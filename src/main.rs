extern crate consalign;
extern crate crossbeam;
extern crate num_cpus;

use consalign::*;
use std::env;

type MeaCssStr = MeaSsStr;

const README_CONTENTS_2: &str = "# consalign.sth\nThis file type contains a predicted RNA structural alignment in the Stockholm format\n\n";

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
    "min_base_pair_prob",
    &format!(
      "A minimum base-pairing probability (Use {DEFAULT_MIN_BPP_ALIGN} by default)"
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_align_prob",
    &format!(
      "A minimum aligning probability (Use {DEFAULT_MIN_ALIGN_PROB_ALIGN} by default)"
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_base_pair_prob_turner",
    &format!(
      "A minimum base-pairing probability for Turner's model (Use {DEFAULT_MIN_BPP_ALIGN_TURNER} by default)"
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_align_prob_turner",
    &format!(
      "A minimum aligning probability for Turner's model (Use {DEFAULT_MIN_ALIGN_PROB_ALIGN_TURNER} by default)"
    ),
    "FLOAT",
  );
  opts.optopt("m", "scoring_model", &format!("Choose a structural alignment scoring model from ensemble, turner, trained (Use {DEFAULT_SCORING_MODEL} by default)"), "STR");
  opts.optopt("u", "train_type", &format!("Choose a scoring parameter training type from trained_transfer, trained_random_init, transferred_only (Use {DEFAULT_TRAIN_TYPE} by default)"), "STR");
  opts.optflag(
    "d",
    "disable_alifold",
    "Disable RNAalifold used in ConsAlifold",
  );
  opts.optflag(
    "p",
    "disable_transplant",
    "Do not transplant trained sequence alignment parameters into Turner's model",
  );
  opts.optopt(
    "t",
    "num_of_threads",
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
  let num_of_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumOfThreads
  };
  let output_dir_path = matches.opt_str("o").unwrap();
  let output_dir_path = Path::new(&output_dir_path);
  let min_bpp = if matches.opt_present("min_base_pair_prob") {
    matches
      .opt_str("min_base_pair_prob")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MIN_BPP_ALIGN
  };
  let min_align_prob = if matches.opt_present("min_align_prob") {
    matches.opt_str("min_align_prob").unwrap().parse().unwrap()
  } else {
    DEFAULT_MIN_ALIGN_PROB_ALIGN
  };
  let min_bpp_turner = if matches.opt_present("min_base_pair_prob_turner") {
    matches
      .opt_str("min_base_pair_prob_turner")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MIN_BPP_ALIGN_TURNER
  };
  let min_align_prob_turner = if matches.opt_present("min_align_prob_turner") {
    matches
      .opt_str("min_align_prob_turner")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MIN_ALIGN_PROB_ALIGN_TURNER
  };
  let scoring_model = if matches.opt_present("m") {
    let scoring_model_str = matches.opt_str("m").unwrap();
    if scoring_model_str == "ensemble" {
      ScoringModel::Ensemble
    } else if scoring_model_str == "turner" {
      ScoringModel::Turner
    } else if scoring_model_str == "trained" {
      ScoringModel::Trained
    } else {
      assert!(false);
      ScoringModel::Ensemble
    }
  } else {
    ScoringModel::Ensemble
  };
  let train_type = if matches.opt_present("u") {
    let train_type_str = matches.opt_str("u").unwrap();
    if train_type_str == "trained_transfer" {
      TrainType::TrainedTransfer
    } else if train_type_str == "trained_random_init" {
      TrainType::TrainedRandomInit
    } else if train_type_str == "transferred_only" {
      TrainType::TransferredOnly
    } else {
      assert!(false);
      TrainType::TrainedTransfer
    }
  } else {
    TrainType::TrainedTransfer
  };
  let disable_alifold = matches.opt_present("d");
  let disable_transplant = matches.opt_present("p");
  let fasta_file_reader = Reader::from_file(Path::new(&input_file_path)).unwrap();
  let mut fasta_records = FastaRecords::new();
  let mut max_seq_len = 0;
  for fasta_record in fasta_file_reader.records() {
    let fasta_record = fasta_record.unwrap();
    let mut seq = convert(fasta_record.seq());
    seq.insert(0, PSEUDO_BASE);
    seq.push(PSEUDO_BASE);
    let seq_len = seq.len();
    if seq_len > max_seq_len {
      max_seq_len = seq_len;
    }
    fasta_records.push(FastaRecord::new(String::from(fasta_record.id()), seq));
  }
  let mut thread_pool = Pool::new(num_of_threads);
  if max_seq_len <= u8::MAX as usize {
    multi_threaded_consalign::<u8, u16>(
      &mut thread_pool,
      &fasta_records,
      output_dir_path,
      input_file_path,
      min_bpp,
      min_align_prob,
      scoring_model,
      train_type,
      disable_alifold,
      min_bpp_turner,
      min_align_prob_turner,
      disable_transplant,
    );
  } else {
    multi_threaded_consalign::<u16, u16>(
      &mut thread_pool,
      &fasta_records,
      output_dir_path,
      input_file_path,
      min_bpp,
      min_align_prob,
      scoring_model,
      train_type,
      disable_alifold,
      min_bpp_turner,
      min_align_prob_turner,
      disable_transplant,
    );
  }
}

fn multi_threaded_consalign<T, U>(
  thread_pool: &mut Pool,
  fasta_records: &FastaRecords,
  output_dir_path: &Path,
  input_file_path: &Path,
  min_bpp: Prob,
  min_align_prob: Prob,
  scoring_model: ScoringModel,
  train_type: TrainType,
  disable_alifold: bool,
  min_bpp_turner: Prob,
  min_align_prob_turner: Prob,
  disable_transplant: bool,
) where
  T: HashIndex,
  U: HashIndex,
{
  let (sa, feature_scores) = wrapped_consalign::<T, U>(
    thread_pool,
    fasta_records,
    output_dir_path,
    input_file_path,
    min_bpp,
    min_align_prob,
    scoring_model,
    train_type,
    disable_alifold,
    min_bpp_turner,
    min_align_prob_turner,
    disable_transplant,
  );
  let output_file_path = output_dir_path.join("consalign.sth");
  write_stockholm_file(&output_file_path, fasta_records, &sa, &feature_scores);
  let mut readme_contents = String::from(README_CONTENTS_2);
  readme_contents.push_str(README_CONTENTS);
  write_readme(output_dir_path, &readme_contents);
}

fn write_stockholm_file<T, U>(
  output_file_path: &Path,
  fasta_records: &FastaRecords,
  sa: &MeaStructAlign<T, U>,
  feature_scores: &FeatureCountsPosterior,
) where
  T: HashIndex,
  U: HashIndex,
{
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = format!(
    "# STOCKHOLM 1.0\n#=GF GA gamma_align={} gamma_basepair={} expected_sps={}\n",
    feature_scores.align_count_posterior, feature_scores.basepair_count_posterior, sa.sps
  );
  let sa_len = sa.struct_align.seq_align.pos_map_sets.len();
  let descriptor = "#=GC SS_cons";
  let descriptor_len = descriptor.len();
  let max_seq_id_len = fasta_records
    .iter()
    .map(|fasta_record| fasta_record.fasta_id.len())
    .max()
    .unwrap();
  let max_seq_id_len = max_seq_id_len.max(descriptor_len);
  let num_of_rnas = sa.rna_ids.len();
  for rna_id in 0..num_of_rnas {
    let seq_id = &fasta_records[rna_id].fasta_id;
    buf_4_writer_2_output_file.push_str(seq_id);
    let mut stockholm_row = vec![b' '; max_seq_id_len - seq_id.len() + 2];
    let fasta_record = &fasta_records[rna_id];
    let seq = &fasta_record.seq;
    let mut sa_row = (0..sa_len)
      .map(|x| {
        let pos_map = sa.struct_align.seq_align.pos_map_sets[x][rna_id]
          .to_usize()
          .unwrap();
        if pos_map == 0 {
          GAP
        } else {
          revert_char(seq[pos_map])
        }
      })
      .collect::<Vec<Char>>();
    stockholm_row.append(&mut sa_row);
    let stockholm_row = unsafe { from_utf8_unchecked(&stockholm_row) };
    buf_4_writer_2_output_file.push_str(stockholm_row);
    buf_4_writer_2_output_file.push('\n');
  }
  buf_4_writer_2_output_file.push_str(descriptor);
  let mut stockholm_row = vec![b' '; max_seq_id_len - descriptor_len + 2];
  let mut mea_css_str = get_mea_css_str(sa, sa_len);
  stockholm_row.append(&mut mea_css_str);
  let stockholm_row = unsafe { from_utf8_unchecked(&stockholm_row) };
  buf_4_writer_2_output_file.push_str(stockholm_row);
  buf_4_writer_2_output_file.push_str("\n//");
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}

fn get_mea_css_str<T, U>(sa: &MeaStructAlign<T, U>, sa_len: usize) -> MeaCssStr
where
  T: HashIndex,
  U: HashIndex,
{
  let mut mea_css_str = vec![UNPAIRING_BASE; sa_len];
  for &(i, j) in &sa.struct_align.bp_pos_pairs {
    mea_css_str[i.to_usize().unwrap()] = BASE_PAIRING_LEFT_BASE;
    mea_css_str[j.to_usize().unwrap()] = BASE_PAIRING_RIGHT_BASE;
  }
  mea_css_str
}
