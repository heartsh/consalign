extern crate consprob;
extern crate consalign;
extern crate crossbeam;

use consalign::*;
use std::env;
use std::io::{BufRead, BufWriter};
use std::fs::File;
use std::fs::create_dir;
use crossbeam::scope;
use std::process::{Command, Output};
use std::fs::remove_file;

type MeaCssStr = MeaSsStr;

const MIN_POW_OF_2: i32 = -4;
const MAX_POW_OF_2: i32 = 10;
const MIN_POW_OF_2_CSS: i32 = MIN_POW_OF_2;
const MAX_POW_OF_2_CSS: i32 = MAX_POW_OF_2;
const DEFAULT_GAMMA: Prob = NEG_INFINITY;
const DEFAULT_MIN_BPP: Prob = 0.01;
const README_CONTENTS_2: &str = "# gamma=x.sth\nThis file type contains a predicted consensus secondary structure in Stockholm format, and this predicted consensus structure is under the prediction accuracy control parameter \"x.\"\n\n";
const MAX_SEQ_LEN_OFFSET: usize = 100;

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
  opts.optopt("g", "gamma", "A specific gamma parameter rather than a range of gamma parameters (for consensus secondary structure prediction)", "FLOAT");
  opts.optopt("", "mix_weight", &format!("A mixture weight for three-way probabilistic consistency (Uses {} by default)", DEFAULT_MIX_WEIGHT), "FLOAT");
  opts.optopt("", "mix_weight_2", &format!("A mixture weight for consensus secondary structure prediction (Uses {} by default)", DEFAULT_MIX_WEIGHT_2), "FLOAT");
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
  let mix_weight_2 = if matches.opt_present("mix_weight_2") {
    matches.opt_str("mix_weight_2").unwrap().parse().unwrap()
  } else {
    DEFAULT_MIX_WEIGHT_2
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
  if max_seq_len + MAX_SEQ_LEN_OFFSET <= u8::MAX as usize {
    multi_threaded_consalign::<u8>(&mut thread_pool, &fasta_records, offset_4_max_gap_num, min_bpp, produces_struct_profs, output_dir_path, gamma, outputs_probs, mix_weight, mix_weight_2, input_file_path);
  } else {
    multi_threaded_consalign::<u16>(&mut thread_pool, &fasta_records, offset_4_max_gap_num, min_bpp, produces_struct_profs, output_dir_path, gamma, outputs_probs, mix_weight, mix_weight_2, input_file_path);
  }
}

fn multi_threaded_consalign<T>(thread_pool: &mut Pool, fasta_records: &FastaRecords, offset_4_max_gap_num: usize, min_bpp: Prob, produces_struct_profs: bool, output_dir_path: &Path, gamma: Prob, outputs_probs: bool, mix_weight: Prob, mix_weight_2: Prob, input_file_path: &Path)
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
  let align_prob_mats_with_rna_id_pairs: ProbMatsWithRnaIdPairs<T> = align_prob_mat_pairs_with_rna_id_pairs.iter().map(|(key, x)| (*key, x.align_prob_mat.clone())).collect();
  let bpp_mats: SparseProbMats<T> = prob_mat_sets.iter().map(|x| x.bpp_mat.clone()).collect();
  compute_and_write_mea_sta(thread_pool, gamma, &output_dir_path, &fasta_records, &align_prob_mats_with_rna_id_pairs, mix_weight_2, &bpp_mats, &input_file_path, offset_4_max_gap_num, min_bpp);
  let mut readme_contents = String::from(README_CONTENTS_2);
  readme_contents.push_str(README_CONTENTS);
  write_readme(output_dir_path, &readme_contents);
}

fn compute_and_write_mea_sta<T>(thread_pool: &mut Pool, gamma: Prob, output_dir_path: &Path, fasta_records: &FastaRecords, align_prob_mats_with_rna_id_pairs: &ProbMatsWithRnaIdPairs<T>, mix_weight_2: Prob, bpp_mats: &SparseProbMats<T>, input_file_path: &Path, offset_4_max_gap_num: usize, min_bpp: Prob)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let feature_score_sets = FeatureCountSetsPosterior::load_trained_score_params();
  let input_file_prefix = input_file_path.file_stem().unwrap().to_str().unwrap();
  let sa_file_path = output_dir_path.join(&format!("{}.aln", input_file_prefix));
  let sa = consalign::<T>(fasta_records, align_prob_mats_with_rna_id_pairs, T::from_usize(offset_4_max_gap_num).unwrap(), bpp_mats, min_bpp, &feature_score_sets, &sa_file_path);
  let mut writer_2_sa_file = BufWriter::new(File::create(sa_file_path.clone()).unwrap());
  let mut buf_4_writer_2_sa_file = format!("CLUSTAL format sequence alignment\n\n");
  let sa_len = sa.cols.len();
  let max_seq_id_len = fasta_records.iter().map(|fasta_record| {fasta_record.fasta_id.len()}).max().unwrap();
  let num_of_rnas = sa.cols[0].len();
  for rna_id in 0 .. num_of_rnas {
    let ref seq_id = fasta_records[rna_id].fasta_id;
    buf_4_writer_2_sa_file.push_str(seq_id);
    let mut clustal_row = vec![' ' as Char; max_seq_id_len - seq_id.len() + 2];
    let mut sa_row = (0 .. sa_len).map(|x| {revert_char(sa.cols[x][rna_id])}).collect::<Vec<Char>>();
    clustal_row.append(&mut sa_row);
    let clustal_row = unsafe {from_utf8_unchecked(&clustal_row)};
    buf_4_writer_2_sa_file.push_str(&clustal_row);
    buf_4_writer_2_sa_file.push_str("\n");
  }
  let _ = writer_2_sa_file.write_all(buf_4_writer_2_sa_file.as_bytes());
  let _ = writer_2_sa_file.flush();
  let sa_file_prefix = sa_file_path.file_stem().unwrap().to_str().unwrap();
  let arg = format!("--id-prefix={}", sa_file_prefix);
  let args = vec!["-p", sa_file_path.to_str().unwrap(), &arg, "--noPS", "--noDP"];
  let _ = run_command("RNAalifold", &args, "Failed to run RNAalifold");
  let mut rnaalifold_bpp_mat = SparseProbMat::<T>::default();
  let cwd = env::current_dir().unwrap();
  let output_file_path = cwd.join(String::from(sa_file_prefix) + "_0001_ali.out");
  let output_file = BufReader::new(File::open(output_file_path.clone()).unwrap());
  for (k, line) in output_file.lines().enumerate() {
    if k == 0 {continue;}
    let line = line.unwrap();
    if !line.starts_with(" ") {continue;}
    let substrings: Vec<&str> = line.split_whitespace().collect();
    let i = T::from_usize(substrings[0].parse().unwrap()).unwrap() - T::one();
    let j = T::from_usize(substrings[1].parse().unwrap()).unwrap() - T::one();
    let mut bpp = String::from(substrings[3]);
    bpp.pop();
    let bpp = 0.01 * bpp.parse::<Prob>().unwrap();
    if bpp == 0. {continue;}
    rnaalifold_bpp_mat.insert((i, j), bpp);
  }
  let _ = remove_file(sa_file_path);
  let _ = remove_file(output_file_path);
  let mix_bpp_mat = get_mix_bpp_mat(bpp_mats, &rnaalifold_bpp_mat, &sa, mix_weight_2);
  if gamma != NEG_INFINITY {
    let output_file_path = output_dir_path.join(&format!("gamma={}.sth", gamma));
    let mea_css = consalifold::<T>(&mix_bpp_mat, gamma, &sa);
    write_stockholm_file(&output_file_path, &sa, &mea_css, fasta_records);
  } else {
    thread_pool.scoped(|scope| {
      for pow_of_2 in MIN_POW_OF_2_CSS .. MAX_POW_OF_2_CSS + 1 {
        let gamma = (2. as Prob).powi(pow_of_2);
        let output_file_path = output_dir_path.join(&format!("gamma={}.sth", gamma));
        let ref ref_2_mix_bpp_mat = mix_bpp_mat;
        let ref ref_2_sa = sa;
        scope.execute(move || {
          let mea_css = consalifold::<T>(ref_2_mix_bpp_mat, gamma, ref_2_sa);
          write_stockholm_file(&output_file_path, ref_2_sa, &mea_css, fasta_records);
        });
      }
    });
  }
}

fn get_mix_bpp_mat<T>(bpp_mats: &SparseProbMats<T>, rnaalifold_bpp_mat: &SparseProbMat<T>, sa: &MeaSeqAlign<T>, mix_weight: Prob) -> ProbMat
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let sa_len = sa.cols.len();
  let num_of_rnas = sa.cols[0].len();
  let mut mix_bpp_mat = vec![vec![0.; sa_len]; sa_len];
  for i in 0 .. sa_len {
    for j in i + 1 .. sa_len {
      let mut mean_bpp = 0.;
      for k in 0 .. num_of_rnas {
        if sa.cols[i][k] == PSEUDO_BASE || sa.cols[j][k] == PSEUDO_BASE {continue;}
        let ref bpp_mat = bpp_mats[k];
        let pos_pair = (sa.pos_map_sets[i][k], sa.pos_map_sets[j][k]);
        match bpp_mat.get(&pos_pair) {
          Some(&bpp) => {
            mean_bpp += bpp;
          }, None => {},
        }
      }
      mix_bpp_mat[i][j] = mix_weight * mean_bpp / num_of_rnas as Prob;
      let pos_pair = (T::from_usize(i).unwrap(), T::from_usize(j).unwrap());
      match rnaalifold_bpp_mat.get(&pos_pair) {
        Some(&rnaalifold_bpp) => {
          mix_bpp_mat[i][j] += (1. - mix_weight) * rnaalifold_bpp;
        }, None => {},
      }
    }
  }
  mix_bpp_mat
}

fn write_stockholm_file<T>(output_file_path: &Path, sa: &MeaSeqAlign<T>, mea_css: &MeaCss<T>, fasta_records: &FastaRecords)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
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
  let descriptor = "#=GC SS_cons";
  let descriptor_len = descriptor.len();
  buf_4_writer_2_output_file.push_str(descriptor);
  let mut stockholm_row = vec![' ' as Char; max_seq_id_len - descriptor_len + 2];
  let mut mea_css_str = get_mea_css_str(&mea_css, sa_len);
  stockholm_row.append(&mut mea_css_str);
  let stockholm_row = unsafe {from_utf8_unchecked(&stockholm_row)};
  buf_4_writer_2_output_file.push_str(&stockholm_row);
  buf_4_writer_2_output_file.push_str("\n//");
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}

fn get_mea_css_str<T>(mea_css: &MeaCss<T>, sa_len: usize) -> MeaCssStr
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Display + Sync + Send,
{
  let mut mea_css_str = vec![UNPAIRING_BASE; sa_len];
  for &(i, j) in &mea_css.bpa_pos_pairs {
    mea_css_str[i.to_usize().unwrap()] = BASE_PAIRING_LEFT_BASE;
    mea_css_str[j.to_usize().unwrap()] = BASE_PAIRING_RIGHT_BASE;
  }
  mea_css_str
}

fn run_command(command: &str, args: &[&str], expect: &str) -> Output {
  Command::new(command).args(args).output().expect(expect)
}
