extern crate consalign;
extern crate crossbeam;
extern crate num_cpus;

use consalign::*;
use std::env;

const README_CONTENTS_CONSALIGN: &str = "# consalign.sth\nThis file type contains a predicted RNA structural alignment in the Stockholm format\n\n";

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_file_path",
    "An input FASTA file path containing RNA sequences to predict their structural alignment",
    "STR",
  );
  opts.reqopt("o", "output_dir_path", "An output directory path", "STR");
  opts.optopt(
    "",
    "min_basepair_prob_trained",
    &format!("A minimum base-pairing probability for the ConsTrain model (Use {DEFAULT_BASEPAIR_PROB_TRAINED} by default)"),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_match_prob_trained",
    &format!("A minimum matching probability for the ConsTrain model (Use {DEFAULT_MATCH_PROB_TRAINED} by default)"),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_basepair_prob_turner",
    &format!(
      "A minimum base-pairing probability for Turner's model (Use {DEFAULT_BASEPAIR_PROB_TURNER} by default)"
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_match_prob_turner",
    &format!(
      "A minimum matching probability for Turner's model (Use {DEFAULT_MATCH_PROB_TURNER} by default)"
    ),
    "FLOAT",
  );
  opts.optopt("m", "score_model", &format!("Choose a structural alignment scoring model from ensemble, turner, trained (Use {DEFAULT_SCORE_MODEL} by default)"), "STR");
  opts.optopt("u", "train_type", &format!("Choose a scoring parameter training type from trained_transfer, trained_randinit, transferred_only (Use {DEFAULT_TRAIN_TYPE} by default)"), "STR");
  opts.optflag(
    "d",
    "disables_alifold",
    "Disable RNAalifold used in ConsAlifold",
  );
  opts.optflag(
    "p",
    "disables_transplant",
    "Do not transplant trained sequence alignment parameters into Turner's model",
  );
  opts.optopt(
    "t",
    "num_threads",
    "The number of threads in multithreading (Use all the threads of this computer by default)",
    "UINT",
  );
  opts.optflag("h", "help", "Print a help menu");
  let matches = match opts.parse(&args[1..]) {
    Ok(opt) => opt,
    Err(failure) => {
      print_program_usage(&program_name, &opts);
      panic!("{}", failure.to_string())
    }
  };
  if matches.opt_present("h") {
    print_program_usage(&program_name, &opts);
    return;
  }
  let input_file_path = matches.opt_str("i").unwrap();
  let input_file_path = Path::new(&input_file_path);
  let num_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumThreads
  };
  let output_dir_path = matches.opt_str("o").unwrap();
  let output_dir_path = Path::new(&output_dir_path);
  let min_basepair_prob_trained = if matches.opt_present("min_basepair_prob_trained") {
    matches
      .opt_str("min_basepair_prob_trained")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_BASEPAIR_PROB_TRAINED
  };
  let min_match_prob_trained = if matches.opt_present("min_match_prob_trained") {
    matches.opt_str("min_match_prob_trained").unwrap().parse().unwrap()
  } else {
    DEFAULT_MATCH_PROB_TRAINED
  };
  let min_basepair_prob_turner = if matches.opt_present("min_basepair_prob_turner") {
    matches
      .opt_str("min_basepair_prob_turner")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_BASEPAIR_PROB_TURNER
  };
  let min_match_prob_turner = if matches.opt_present("min_match_prob_turner") {
    matches
      .opt_str("min_match_prob_turner")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MATCH_PROB_TURNER
  };
  let score_model = if matches.opt_present("m") {
    let x = matches.opt_str("m").unwrap();
    if x == "ensemble" {
      ScoreModel::Ensemble
    } else if x == "turner" {
      ScoreModel::Turner
    } else if x == "trained" {
      ScoreModel::Trained
    } else {
      panic!();
    }
  } else {
    ScoreModel::Ensemble
  };
  let train_type = if matches.opt_present("u") {
    let x = matches.opt_str("u").unwrap();
    if x == "trained_transfer" {
      TrainType::TrainedTransfer
    } else if x == "trained_random_init" {
      TrainType::TrainedRandinit
    } else if x == "transferred_only" {
      TrainType::TransferredOnly
    } else {
      panic!();
    }
  } else {
    TrainType::TrainedTransfer
  };
  let disables_alifold = matches.opt_present("d");
  let disables_transplant = matches.opt_present("p");
  let fasta_file_reader = Reader::from_file(Path::new(&input_file_path)).unwrap();
  let mut fasta_records = FastaRecords::new();
  let mut max_seq_len = 0;
  for x in fasta_file_reader.records() {
    let x = x.unwrap();
    let mut y = bytes2seq(x.seq());
    y.insert(0, PSEUDO_BASE);
    y.push(PSEUDO_BASE);
    let z = y.len();
    if z > max_seq_len {
      max_seq_len = z;
    }
    fasta_records.push(FastaRecord::new(String::from(x.id()), y));
  }
  let mut thread_pool = Pool::new(num_threads);
  let inputs = (
    &mut thread_pool,
    &fasta_records,
    output_dir_path,
    input_file_path,
    min_basepair_prob_trained,
    min_match_prob_trained,
    score_model,
    train_type,
    disables_alifold,
    min_basepair_prob_turner,
    min_match_prob_turner,
    disables_transplant,
  );
  if max_seq_len <= u8::MAX as usize {
    consalign_multithreaded::<u8, u16>(inputs);
  } else {
    consalign_multithreaded::<u16, u16>(inputs);
  }
}

fn consalign_multithreaded<T, U>(inputs: InputsConsalignWrapped)
where
  T: HashIndex,
  U: HashIndex,
{
  let (_, x, y, _, _, _, _, _, _, _, _, _) = inputs;
  let (z, a) = consalign_wrapped::<T, U>(inputs);
  let b = y.join("consalign.sth");
  write_stockholm_file(&b, x, &z, &a);
  let mut a = String::from(README_CONTENTS_CONSALIGN);
  a.push_str(README_CONTENTS);
  write_readme(y, &a);
}

fn write_stockholm_file<T, U>(
  output_file_path: &Path,
  fasta_records: &FastaRecords,
  alignfold: &AlignfoldWrapped<T, U>,
  alignfold_hyperparams: &AlignfoldHyperparams,
) where
  T: HashIndex,
  U: HashIndex,
{
  let mut writer = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf = format!(
    "# STOCKHOLM 1.0\n#=GF GA hyperparam_match={} hyperparam_basepair={} expected_accuracy={}\n",
    alignfold_hyperparams.param_match, alignfold_hyperparams.param_basepair, alignfold.accuracy
  );
  let align_len = alignfold.alignfold.align.pos_map_sets.len();
  let descriptor = "#=GC SS_cons";
  let descriptor_len = descriptor.len();
  let max_seq_id_len = fasta_records
    .iter()
    .map(|x| x.fasta_id.len())
    .max()
    .unwrap();
  let max_seq_id_len = max_seq_id_len.max(descriptor_len);
  for (x, y) in fasta_records.iter().enumerate() {
    let z = &y.fasta_id;
    buf.push_str(z);
    let mut a = vec![b' '; max_seq_id_len - z.len() + 2];
    let y = &y.seq;
    let mut z = (0..align_len)
      .map(|z| {
        let z = alignfold.alignfold.align.pos_map_sets[z][x]
          .to_usize()
          .unwrap();
        if z == 0 {
          GAP
        } else {
          base2char(y[z])
        }
      })
      .collect::<Vec<Char>>();
    a.append(&mut z);
    let a = unsafe { from_utf8_unchecked(&a) };
    buf.push_str(a);
    buf.push('\n');
  }
  buf.push_str(descriptor);
  let mut stockholm_row = vec![b' '; max_seq_id_len - descriptor_len + 2];
  let mut fold_str = get_fold_str(alignfold, /* align_len */);
  stockholm_row.append(&mut fold_str);
  let stockholm_row = unsafe { from_utf8_unchecked(&stockholm_row) };
  buf.push_str(stockholm_row);
  buf.push_str("\n//");
  let _ = writer.write_all(buf.as_bytes());
}

fn get_fold_str<T, U>(x: &AlignfoldWrapped <T, U>) -> FoldStr
where
  T: HashIndex,
  U: HashIndex,
{
  let mut y = vec![UNPAIR; x.alignfold.align.pos_map_sets.len()];
  for &(i, j) in &x.alignfold.basepairs {
    y[i.to_usize().unwrap()] = BASEPAIR_LEFT;
    y[j.to_usize().unwrap()] = BASEPAIR_RIGHT;
  }
  y
}
