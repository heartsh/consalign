extern crate consprob;
extern crate consalign;

use consalign::*;
use std::env;

const MIN_POW_OF_2: i32 = -4;
const MAX_POW_OF_2: i32 = 10;
const DEFAULT_GAMMA: Prob = NEG_INFINITY;
const README_CONTENTS_2: &str = "# gamma=x.sth\nThis file type contains a predicted consensus secondary structure in Stockholm format, and this predicted consensus structure is under the prediction accuracy control parameter \"x.\"\n\n";

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_file_path",
    "A path to an input FASTA file containing RNA sequences to predict probabilities",
    "STR",
  );
  opts.reqopt(
    "o",
    "output_dir_path",
    "A path to an output directory",
    "STR",
  );
  opts.optopt(
    "",
    "min_base_pair_prob",
    &format!(
      "A minimum base-pairing-probability (Uses {} by default)",
      DEFAULT_MIN_BPP
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "offset_4_max_gap_num",
    &format!(
      "An offset for maximum numbers of gaps (Uses {} by default)",
      DEFAULT_OFFSET_4_MAX_GAP_NUM
    ),
    "UINT",
  );
  opts.optopt("g", "gamma", "A specific gamma parameter rather than a range of gamma parameters", "FLOAT");
  opts.optopt("", "mix_weight", &format!("A mixture weight (Uses {} by default)", DEFAULT_MIX_WEIGHT), "FLOAT");
  opts.optopt("t", "num_of_threads", "The number of threads in multithreading (Uses the number of the threads of this computer by default)", "UINT");
  opts.optflag(
    "s",
    "produces_struct_profs",
    &format!("Also compute RNA structural context profiles"),
  );
  opts.optflag("p", "outputs_probs", &format!("Output probabilities"));
  opts.optflag("h", "help", "Print a help menu");
  let matches = match opts.parse(&args[1..]) {
    Ok(opt) => opt,
    Err(failure) => {
      print_program_usage(&program_name, &opts);
      panic!(failure.to_string())
    }
  };
  if matches.opt_present("h") {
    print_program_usage(&program_name, &opts);
    return;
  }
  let input_file_path = matches.opt_str("i").unwrap();
  let input_file_path = Path::new(&input_file_path);
  let min_bpp = if matches.opt_present("min_base_pair_prob") {
    matches
      .opt_str("min_base_pair_prob")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MIN_BPP
  };
  let offset_4_max_gap_num = if matches.opt_present("offset_4_max_gap_num") {
    matches
      .opt_str("offset_4_max_gap_num")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_OFFSET_4_MAX_GAP_NUM
  };
  let gamma = if matches.opt_present("gamma") {
    matches.opt_str("gamma").unwrap().parse().unwrap()
  } else {
    DEFAULT_GAMMA
  };
  let num_of_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumOfThreads
  };
  let produces_struct_profs = matches.opt_present("s");
  let outputs_probs = matches.opt_present("p") || produces_struct_profs;
  let mix_weight = if matches.opt_present("mix_weight") {
    matches.opt_str("mix_weight").unwrap().parse().unwrap()
  } else {
    DEFAULT_MIX_WEIGHT
  };
  let output_dir_path = matches.opt_str("o").unwrap();
  let output_dir_path = Path::new(&output_dir_path);
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
    multi_threaded_consalign::<u8>(&mut thread_pool, &fasta_records, offset_4_max_gap_num, min_bpp, produces_struct_profs, output_dir_path, gamma, outputs_probs, mix_weight);
  } else {
    multi_threaded_consalign::<u16>(&mut thread_pool, &fasta_records, offset_4_max_gap_num, min_bpp, produces_struct_profs, output_dir_path, gamma, outputs_probs, mix_weight);
  }
}

fn multi_threaded_consalign<T>(thread_pool: &mut Pool, fasta_records: &FastaRecords, offset_4_max_gap_num: usize, min_bpp: Prob, produces_struct_profs: bool, output_dir_path: &Path, gamma: Prob, outputs_probs: bool, mix_weight: Prob)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let (prob_mat_sets, align_prob_mat_pairs_with_rna_id_pairs) = consprob::<T>(thread_pool, fasta_records, min_bpp, T::from_usize(offset_4_max_gap_num).unwrap(), produces_struct_profs, true, mix_weight);
  if !output_dir_path.exists() {
    let _ = create_dir(output_dir_path);
  }
  if outputs_probs {
    write_prob_mat_sets::<T>(output_dir_path, &prob_mat_sets, produces_struct_profs, &align_prob_mat_pairs_with_rna_id_pairs, true);
  }
  println!("Start");
  if gamma != NEG_INFINITY {
    let output_file_path = output_dir_path.join(&format!("gamma={}.sth", gamma));
    compute_and_write_mea_sta(gamma, &output_file_path, &fasta_records, &align_prob_mat_pairs_with_rna_id_pairs);
  } else {
    thread_pool.scoped(|scope| {
      for pow_of_2 in MIN_POW_OF_2 .. MAX_POW_OF_2 + 1 {
        let gamma = (2. as Prob).powi(pow_of_2);
        let ref ref_2_align_prob_mat_pairs_with_rna_id_pairs = align_prob_mat_pairs_with_rna_id_pairs;
        let output_file_path = output_dir_path.join(&format!("gamma={}.sth", gamma));
        scope.execute(move || {
          compute_and_write_mea_sta::<T>(gamma, &output_file_path, fasta_records, ref_2_align_prob_mat_pairs_with_rna_id_pairs);
        });
      }
    });
  }
  let mut readme_contents = String::from(README_CONTENTS_2);
  readme_contents.push_str(README_CONTENTS);
  write_readme(output_dir_path, &readme_contents);
}

fn compute_and_write_mea_sta<T>(gamma: Prob, output_file_path: &Path, fasta_records: &FastaRecords, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let sa = consalign::<T>(fasta_records, gamma, align_prob_mat_pairs_with_rna_id_pairs);
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = format!("# STOCKHOLM 1.0\n");
  let sa_len = sa.cols.len();
  let max_seq_id_len = fasta_records.iter().map(|fasta_record| {fasta_record.fasta_id.len()}).max().unwrap();
  let num_of_rnas = sa.cols[0].len();
  for rna_id in 0 .. num_of_rnas {
    let ref seq_id = fasta_records[rna_id].fasta_id;
    buf_4_writer_2_output_file.push_str(seq_id);
    let mut stockholm_row = vec![' ' as Char; max_seq_id_len - seq_id.len() + 2];
    let mut sa_row = (0 .. sa_len).map(|x| {revert_char(sa.cols[x][rna_id])}).collect::<Vec<Char>>();
    stockholm_row.append(&mut sa_row);
    let stockholm_row = unsafe {from_utf8_unchecked(&stockholm_row)};
    buf_4_writer_2_output_file.push_str(&stockholm_row);
    buf_4_writer_2_output_file.push_str("\n");
  }
  /* let descriptor = "#=GC SS_cons";
  let descriptor_len = descriptor.len();
  buf_4_writer_2_output_file.push_str(descriptor);
  let mut stockholm_row = vec![' ' as Char; max_seq_id_len - descriptor_len + 2];
  let mut mea_css_str = get_mea_css_str(&mea_css, sa_len);
  stockholm_row.append(&mut mea_css_str);
  let stockholm_row = unsafe {from_utf8_unchecked(&stockholm_row)};
  buf_4_writer_2_output_file.push_str(&stockholm_row);
  buf_4_writer_2_output_file.push_str("\n//"); */
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}
