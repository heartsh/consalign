extern crate consprob;
extern crate consalign;
extern crate crossbeam;

use consalign::*;
use std::env;
use std::fs::File;
use std::fs::create_dir;

type MeaCssStr = MeaSsStr;

const README_CONTENTS_2: &str = "# consalign.sth\nThis file type contains a predicted RNA structural alignment in the Stockholm format\n\n";
const MIX_COEFFS: [Prob; 11] = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

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
      DEFAULT_MIN_BPP_ALIGN
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "offset_4_max_gap_num",
    &format!(
      "An offset for maximum numbers of gaps (Uses {} by default)",
      DEFAULT_OFFSET_4_MAX_GAP_NUM_ALIGN
    ),
    "UINT",
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
  let min_bpp = if matches.opt_present("min_base_pair_prob") {
    matches
      .opt_str("min_base_pair_prob")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MIN_BPP_ALIGN
  };
  let offset_4_max_gap_num = if matches.opt_present("offset_4_max_gap_num") {
    matches
      .opt_str("offset_4_max_gap_num")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_OFFSET_4_MAX_GAP_NUM_ALIGN
  };
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
  multi_threaded_consalign::<u16>(&mut thread_pool, &fasta_records, offset_4_max_gap_num, min_bpp, produces_struct_profs, output_dir_path, outputs_probs, input_file_path);
}

fn multi_threaded_consalign<T>(thread_pool: &mut Pool, fasta_records: &FastaRecords, offset_4_max_gap_num: usize, min_bpp: Prob, produces_struct_profs: bool, output_dir_path: &Path, outputs_probs: bool, input_file_path: &Path)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let (prob_mat_sets, align_prob_mat_pairs_with_rna_id_pairs) = consprob::<T>(thread_pool, fasta_records, min_bpp, T::from_usize(offset_4_max_gap_num).unwrap(), produces_struct_profs, true);
  if !output_dir_path.exists() {
    let _ = create_dir(output_dir_path);
  }
  if outputs_probs {
    write_prob_mat_sets::<T>(output_dir_path, &prob_mat_sets, produces_struct_profs, &align_prob_mat_pairs_with_rna_id_pairs, true);
  }
  let align_prob_mats_with_rna_id_pairs: ProbMatsWithRnaIdPairs<T> = align_prob_mat_pairs_with_rna_id_pairs.iter().map(|(key, x)| (*key, x.align_prob_mat.clone())).collect();
  let bpp_mats: SparseProbMats<T> = prob_mat_sets.iter().map(|x| x.bpp_mat.clone()).collect();
  let input_file_prefix = input_file_path.file_stem().unwrap().to_str().unwrap();
  let mut candidates = Vec::new();
  for log_gamma_basepair in MIN_LOG_GAMMA_BASEPAIR .. MAX_LOG_GAMMA_BASEPAIR + 1 {
    let basepair_count_posterior = (2. as Prob).powi(log_gamma_basepair) + 1.;
    for log_gamma_align in MIN_LOG_GAMMA_ALIGN .. MAX_LOG_GAMMA_ALIGN + 1 {
      let align_count_posterior = (2. as Prob).powi(log_gamma_align) + 1.;
      let mut feature_scores = FeatureCountsPosterior::new(0.);
      feature_scores.basepair_count_posterior = basepair_count_posterior;
      feature_scores.align_count_posterior = align_count_posterior;
      candidates.push((feature_scores, MeaStructAlign::<T>::new()));
    }
  }
  thread_pool.scoped(|scope| {
    let ref ref_2_align_prob_mats_with_rna_id_pairs = align_prob_mats_with_rna_id_pairs;
    let ref ref_2_bpp_mats = bpp_mats;
    for candidate in &mut candidates {
      let sa_file_path = output_dir_path.join(&format!("{}_g1={}_g2={}.aln", input_file_prefix, candidate.0.basepair_count_posterior, candidate.0.align_count_posterior));
      scope.execute(move || {
        candidate.1 = consalign::<T>(fasta_records, ref_2_align_prob_mats_with_rna_id_pairs, ref_2_bpp_mats, &candidate.0, &sa_file_path);
      })
    }
  });
  let mut max_acc = NEG_INFINITY;
  let mut argmax_params = FeatureCountsPosterior::new(NEG_INFINITY);
  let mut argmax_align = MeaStructAlign::<T>::new();
  for candidate in &candidates {
    if candidate.1.acc > max_acc {
      argmax_params = candidate.0.clone();
      argmax_align = candidate.1.clone();
      max_acc = candidate.1.acc;
    }
  }
  let sa_file_path = output_dir_path.join(&format!("{}_g1={}_g2={}.aln", input_file_prefix, argmax_params.basepair_count_posterior, argmax_params.align_count_posterior));
  let bpp_mat_alifold = get_bpp_mat_alifold(&argmax_align, &sa_file_path, fasta_records);
  let mut candidates = Vec::new();
  for log_gamma_basepair in MIN_LOG_GAMMA_BASEPAIR .. MAX_LOG_GAMMA_BASEPAIR + 1 {
    let basepair_count_posterior = (2. as Prob).powi(log_gamma_basepair) + 1.;
    // for mix_coeff in 0 .. 11 {
    for &mix_coeff in &MIX_COEFFS {
      // let mix_coeff = 0.1 * mix_coeff as Prob;
      candidates.push((basepair_count_posterior, mix_coeff, SparsePosMat::<T>::default(), NEG_INFINITY));
    }
  }
  thread_pool.scoped(|scope| {
    let ref ref_2_bpp_mats = bpp_mats;
    let ref ref_2_bpp_mat_alifold = bpp_mat_alifold;
    let ref ref_2_argmax_align = argmax_align;
    let ref ref_2_fasta_records = fasta_records;
    for candidate in &mut candidates {
      scope.execute(move || {
        let result = consalifold(ref_2_bpp_mats, ref_2_bpp_mat_alifold, ref_2_argmax_align, candidate.0, candidate.1, ref_2_fasta_records);
        candidate.2 = result.0;
        candidate.3 = result.1;
      })
    }
  });
  let mut max_acc = NEG_INFINITY;
  let mut argmax_params_2 = (NEG_INFINITY, NEG_INFINITY);
  let mut argmax_css = SparsePosMat::<T>::default();
  for candidate in &candidates {
    if candidate.3 > max_acc {
      argmax_params_2 = (candidate.0, candidate.1);
      argmax_css = candidate.2.clone();
      max_acc = candidate.3;
    }
  }
  argmax_align.bp_col_pairs = argmax_css;
  argmax_align.sort();
  compute_and_write_mea_sta(&output_dir_path, fasta_records, &argmax_params, &argmax_params_2, &argmax_align);
  let mut readme_contents = String::from(README_CONTENTS_2);
  readme_contents.push_str(README_CONTENTS);
  write_readme(output_dir_path, &readme_contents);
}

fn compute_and_write_mea_sta<T>(output_dir_path: &Path, fasta_records: &FastaRecords, feature_scores: &FeatureCountsPosterior, argmax_params_2: &(FeatureCount, FeatureCount), sa: &MeaStructAlign<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let output_file_path = output_dir_path.join(&format!("consalign_g1={}_g2={}_g3={}_t={}.sth", feature_scores.basepair_count_posterior, feature_scores.align_count_posterior, argmax_params_2.0, argmax_params_2.1));
  write_stockholm_file(&output_file_path, &sa, fasta_records);
}

fn write_stockholm_file<T>(output_file_path: &Path, sa: &MeaStructAlign<T>, fasta_records: &FastaRecords)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = format!("# STOCKHOLM 1.0\n");
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
