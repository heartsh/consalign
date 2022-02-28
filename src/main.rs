extern crate consprob;
extern crate consalign;
extern crate crossbeam;

use consalign::*;
use std::env;
use std::fs::File;
use std::fs::create_dir;

type MeaCssStr = MeaSsStr;

const README_CONTENTS_2: &str = "# consalign.sth\nThis file type contains a predicted RNA structural alignment in the Stockholm format\n\n";

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
  let num_of_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumOfThreads
  };
  let produces_struct_profs = matches.opt_present("s");
  let outputs_probs = matches.opt_present("p") || produces_struct_profs;
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
  multi_threaded_consalign::<u16>(&mut thread_pool, &fasta_records, output_dir_path, input_file_path);
}

fn multi_threaded_consalign<T>(thread_pool: &mut Pool, fasta_records: &FastaRecords, output_dir_path: &Path, input_file_path: &Path)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let num_of_fasta_records = fasta_records.len();
  let mut bpp_mats = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  thread_pool.scoped(|scope| {
    for (bpp_mat, fasta_record) in multizip((bpp_mats.iter_mut(), fasta_records.iter())) {
      let seq_len = fasta_record.seq.len();
      scope.execute(move || {
        let (obtained_bpp_mat, _) = mccaskill_algo(&fasta_record.seq[1 .. seq_len - 1], true);
        *bpp_mat = remove_small_bpps_from_bpp_mat::<T>(&obtained_bpp_mat, 0.);
      });
    }
  });
  let mut align_prob_mats_with_rna_id_pairs = ProbMatsWithRnaIdPairs::default();
  let mut insert_prob_set_pairs_with_rna_id_pairs = ProbSetPairsWithRnaIdPairs::default();
  for rna_id_1 in 0 .. num_of_fasta_records {
    for rna_id_2 in rna_id_1 + 1 .. num_of_fasta_records {
      let rna_id_pair = (rna_id_1, rna_id_2);
      align_prob_mats_with_rna_id_pairs.insert(rna_id_pair, ProbMat::new());
      insert_prob_set_pairs_with_rna_id_pairs.insert(rna_id_pair, (Probs::new(), Probs::new()));
    }
  }
  thread_pool.scoped(|scope| {
    for (rna_id_pair, align_prob_mat) in align_prob_mats_with_rna_id_pairs.iter_mut() {
      let seq_pair = (&fasta_records[rna_id_pair.0].seq[..], &fasta_records[rna_id_pair.1].seq[..]);
      scope.execute(move || {
        *align_prob_mat = durbin_algo(&seq_pair);
      });
    }
  });
  thread_pool.scoped(|scope| {
    for (rna_id_pair, insert_prob_set_pair) in insert_prob_set_pairs_with_rna_id_pairs.iter_mut() {
      let ref align_prob_mat = align_prob_mats_with_rna_id_pairs[rna_id_pair];
      scope.execute(move || {
        *insert_prob_set_pair = get_insert_prob_set_pair(align_prob_mat);
      });
    }
  });
  if !output_dir_path.exists() {
    let _ = create_dir(output_dir_path);
  }
  let input_file_prefix = input_file_path.file_stem().unwrap().to_str().unwrap();
  let sa_file_path = output_dir_path.join(&format!("{}.aln", input_file_prefix));
  let mut candidates = Vec::new();
  for log_gamma_align in MIN_LOG_GAMMA_ALIGN .. MAX_LOG_GAMMA_ALIGN + 1 {
    let align_count_posterior = (2. as Prob).powi(log_gamma_align) + 1.;
    for log_gamma_basepair in MIN_LOG_GAMMA_BASEPAIR .. MAX_LOG_GAMMA_BASEPAIR + 1 {
      let basepair_count_posterior = (2. as Prob).powi(log_gamma_basepair) + 1.;
      let mut feature_scores = FeatureCountsPosterior::new(align_count_posterior);
      feature_scores.basepair_count_posterior = basepair_count_posterior;
      candidates.push((feature_scores, MeaStructAlign::new()));
    }
  }
  thread_pool.scoped(|scope| {
    let ref ref_2_align_prob_mats_with_rna_id_pairs = align_prob_mats_with_rna_id_pairs;
    let ref ref_2_bpp_mats = bpp_mats;
    let ref ref_2_insert_prob_set_pairs_with_rna_id_pairs = insert_prob_set_pairs_with_rna_id_pairs;
    for candidate in &mut candidates {
      scope.execute(move || {
        candidate.1 = consalign::<T>(fasta_records, ref_2_align_prob_mats_with_rna_id_pairs, ref_2_bpp_mats, &candidate.0, ref_2_insert_prob_set_pairs_with_rna_id_pairs, &mut Pool::new(1));
      });
    }
  });
  let mut sa = MeaStructAlign::new();
  let mut feature_scores = FeatureCountsPosterior::new(0.);
  for candidate in &candidates {
    let ref tmp_sa = candidate.1;
    if tmp_sa.acc > sa.acc {
      feature_scores = candidate.0.clone();
      sa = tmp_sa.clone();
    }
  }
  let bpp_mat_alifold = get_bpp_mat_alifold(&sa, &sa_file_path, fasta_records);
  let mix_bpp_mat = get_mix_bpp_mat(&sa, &bpp_mats, &bpp_mat_alifold);
  sa.bp_col_pairs = consalifold(&mix_bpp_mat, &sa, BASEPAIR_COUNT_POSTERIOR_ALIFOLD, &fasta_records);
  sa.sort();
  let output_file_path = output_dir_path.join(&format!("consalign.sth"));
  write_stockholm_file(&output_file_path, fasta_records, &sa, &feature_scores);
  let mut readme_contents = String::from(README_CONTENTS_2);
  readme_contents.push_str(README_CONTENTS);
  write_readme(output_dir_path, &readme_contents);
}

fn write_stockholm_file<T>(output_file_path: &Path, fasta_records: &FastaRecords, sa: &MeaStructAlign<T>, feature_scores: &FeatureCountsPosterior)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = format!("# STOCKHOLM 1.0\n#=GF GA gamma_align={} gamma_basepair={}\n", feature_scores.align_count_posterior, feature_scores.basepair_count_posterior);
  let sa_len = sa.cols.len();
  let descriptor = "#=GC SS_cons";
  let descriptor_len = descriptor.len();
  let max_seq_id_len = fasta_records.iter().map(|fasta_record| {fasta_record.fasta_id.len()}).max().unwrap();
  let max_seq_id_len = max_seq_id_len.max(descriptor_len);
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
  buf_4_writer_2_output_file.push_str(descriptor);
  let mut stockholm_row = vec![' ' as Char; max_seq_id_len - descriptor_len + 2];
  let mut mea_css_str = get_mea_css_str(&sa, sa_len);
  stockholm_row.append(&mut mea_css_str);
  let stockholm_row = unsafe {from_utf8_unchecked(&stockholm_row)};
  buf_4_writer_2_output_file.push_str(&stockholm_row);
  buf_4_writer_2_output_file.push_str("\n//");
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}

fn get_mea_css_str<T>(sa: &MeaStructAlign<T>, sa_len: usize) -> MeaCssStr
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let mut mea_css_str = vec![UNPAIRING_BASE; sa_len];
  for &(i, j) in &sa.bp_col_pairs {
    mea_css_str[i.to_usize().unwrap()] = BASE_PAIRING_LEFT_BASE;
    mea_css_str[j.to_usize().unwrap()] = BASE_PAIRING_RIGHT_BASE;
  }
  mea_css_str
}
