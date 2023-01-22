extern crate consalign;
extern crate num_cpus;
extern crate crossbeam;

use consalign::*;
use std::env;
use std::fs::create_dir;
use std::fs::File;
use std::mem::drop;

type MeaCssStr = MeaSsStr;

const README_CONTENTS_2: &str = "# consalign.sth\nThis file type contains a predicted RNA structural alignment in the Stockholm format\n\n";
enum ScoringModel {
  Ensemble,
  Turner,
  Trained,
}
const DEFAULT_SCORING_MODEL: &str = "ensemble";

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
      "A minimum base-pairing probability (Use {} by default)",
      DEFAULT_MIN_BPP_ALIGN
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_align_prob",
    &format!(
      "A minimum aligning probability (Use {} by default)",
      DEFAULT_MIN_ALIGN_PROB_ALIGN
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_base_pair_prob_turner",
    &format!(
      "A minimum base-pairing probability for Turner's model (Use {} by default)",
      DEFAULT_MIN_BPP_ALIGN_TURNER
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_align_prob_turner",
    &format!(
      "A minimum aligning probability for Turner's model (Use {} by default)",
      DEFAULT_MIN_ALIGN_PROB_ALIGN_TURNER
    ),
    "FLOAT",
  );
  opts.optopt("m", "scoring_model", &format!("Choose a structural alignment scoring model from ensemble, turner, trained (Use {} by default)", DEFAULT_SCORING_MODEL), "STR");
  opts.optopt("u", "train_type", &format!("Choose a scoring parameter training type from trained_transfer, trained_random_init, transferred_only (Use {} by default)", DEFAULT_TRAIN_TYPE), "STR");
  opts.optflag(
    "d",
    "disable_alifold",
    &format!("Disable RNAalifold used in ConsAlifold"),
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
  let mut align_feature_score_sets = AlignFeatureCountSets::new(0.);
  if disable_transplant {
    align_feature_score_sets.transfer();
  } else {
    copy_feature_count_sets_align(
      &mut align_feature_score_sets,
      &FeatureCountSets::load_trained_score_params(),
    );
  }
  let seqs = fasta_records.iter().map(|x| &x.seq[..]).collect();
  let (prob_mat_sets_turner, align_prob_mat_pairs_with_rna_id_pairs_turner) =
    if matches!(scoring_model, ScoringModel::Ensemble)
      || matches!(scoring_model, ScoringModel::Turner)
    {
      consprob::<T>(
        thread_pool,
        &seqs,
        min_bpp_turner,
        min_align_prob_turner,
        false,
        true,
        &align_feature_score_sets,
      )
    } else {
      (
        ProbMatSets::<T>::default(),
        AlignProbMatSetsWithRnaIdPairs::<T>::default(),
      )
    };
  let align_prob_mats_with_rna_id_pairs_turner: SparseProbMatsWithRnaIdPairs<T> =
    align_prob_mat_pairs_with_rna_id_pairs_turner
      .iter()
      .map(|(key, x)| (*key, x.align_prob_mat.clone()))
      .collect();
  let bpp_mats_turner: SparseProbMats<T> = prob_mat_sets_turner
    .iter()
    .map(|x| x.bpp_mat.clone())
    .collect();
  drop(prob_mat_sets_turner);
  drop(align_prob_mat_pairs_with_rna_id_pairs_turner);
  let (prob_mat_sets_trained, align_prob_mat_pairs_with_rna_id_pairs_trained) =
    if matches!(scoring_model, ScoringModel::Ensemble)
      || matches!(scoring_model, ScoringModel::Trained)
    {
      consprob_trained::<T>(
        thread_pool,
        &seqs,
        min_bpp,
        min_align_prob,
        false,
        true,
        train_type,
      )
    } else {
      (
        ProbMatSets::<T>::default(),
        AlignProbMatSetsWithRnaIdPairs::<T>::default(),
      )
    };
  let align_prob_mats_with_rna_id_pairs_trained: SparseProbMatsWithRnaIdPairs<T> =
    align_prob_mat_pairs_with_rna_id_pairs_trained
      .iter()
      .map(|(key, x)| (*key, x.align_prob_mat.clone()))
      .collect();
  let bpp_mats_trained: SparseProbMats<T> = prob_mat_sets_trained
    .iter()
    .map(|x| x.bpp_mat.clone())
    .collect();
  drop(prob_mat_sets_trained);
  drop(align_prob_mat_pairs_with_rna_id_pairs_trained);
  let num_of_fasta_records = fasta_records.len();
  let mut bpp_mats_fused = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  let mut align_prob_mats_with_rna_id_pairs_fused = SparseProbMatsWithRnaIdPairs::<T>::default();
  let mut insert_prob_set_pairs_with_rna_id_pairs = ProbSetPairsWithRnaIdPairs::default();
  for rna_id_1 in 0..num_of_fasta_records {
    for rna_id_2 in rna_id_1 + 1..num_of_fasta_records {
      let rna_id_pair = (rna_id_1, rna_id_2);
      align_prob_mats_with_rna_id_pairs_fused.insert(rna_id_pair, SparseProbMat::<T>::default());
      insert_prob_set_pairs_with_rna_id_pairs.insert(rna_id_pair, (Probs::new(), Probs::new()));
    }
  }
  if matches!(scoring_model, ScoringModel::Ensemble) {
    thread_pool.scoped(|scope| {
      for (bpp_mat_fused, bpp_mat_turner, bpp_mat_trained) in multizip((
        bpp_mats_fused.iter_mut(),
        bpp_mats_turner.iter(),
        bpp_mats_trained.iter(),
      )) {
        scope.execute(move || {
          *bpp_mat_fused = bpp_mat_turner
            .iter()
            .map(|(pos_pair, &bpp)| (*pos_pair, 0.5 * bpp))
            .collect();
          for (pos_pair, bpp_trained) in bpp_mat_trained {
            let bpp_trained = 0.5 * bpp_trained;
            match bpp_mat_fused.get_mut(pos_pair) {
              Some(bpp_fused) => {
                *bpp_fused += bpp_trained;
              }
              None => {
                bpp_mat_fused.insert(*pos_pair, bpp_trained);
              }
            }
          }
        });
      }
    });
    thread_pool.scoped(|scope| {
      for (rna_id_pair, align_prob_mat_fused) in align_prob_mats_with_rna_id_pairs_fused.iter_mut()
      {
        let ref align_prob_mat_turner = align_prob_mats_with_rna_id_pairs_turner[rna_id_pair];
        let ref align_prob_mat_trained = align_prob_mats_with_rna_id_pairs_trained[rna_id_pair];
        scope.execute(move || {
          *align_prob_mat_fused = align_prob_mat_turner
            .iter()
            .map(|(pos_pair, &align_prob)| (*pos_pair, 0.5 * align_prob))
            .collect();
          for (pos_pair, align_prob_trained) in align_prob_mat_trained {
            let align_prob_trained = 0.5 * align_prob_trained;
            match align_prob_mat_fused.get_mut(pos_pair) {
              Some(align_prob_fused) => {
                *align_prob_fused += align_prob_trained;
              }
              None => {
                align_prob_mat_fused.insert(*pos_pair, align_prob_trained);
              }
            }
          }
        });
      }
    });
  } else if matches!(scoring_model, ScoringModel::Turner) {
    bpp_mats_fused = bpp_mats_turner.clone();
    align_prob_mats_with_rna_id_pairs_fused = align_prob_mats_with_rna_id_pairs_turner.clone();
  } else {
    bpp_mats_fused = bpp_mats_trained.clone();
    align_prob_mats_with_rna_id_pairs_fused = align_prob_mats_with_rna_id_pairs_trained.clone();
  }
  thread_pool.scoped(|scope| {
    for (rna_id_pair, insert_prob_set_pair) in insert_prob_set_pairs_with_rna_id_pairs.iter_mut() {
      let ref align_prob_mat = align_prob_mats_with_rna_id_pairs_fused[rna_id_pair];
      let seq_len_pair = (
        fasta_records[rna_id_pair.0].seq.len(),
        fasta_records[rna_id_pair.1].seq.len(),
      );
      scope.execute(move || {
        *insert_prob_set_pair = get_insert_prob_set_pair(align_prob_mat, &seq_len_pair);
      });
    }
  });
  if !output_dir_path.exists() {
    let _ = create_dir(output_dir_path);
  }
  let input_file_prefix = input_file_path.file_stem().unwrap().to_str().unwrap();
  let sa_file_path = output_dir_path.join(&format!("{}.aln", input_file_prefix));
  let mut candidates = Vec::new();
  for log_gamma_align in MIN_LOG_GAMMA_ALIGN..MAX_LOG_GAMMA_ALIGN + 1 {
    let align_count_posterior = (2. as Prob).powi(log_gamma_align) + 1.;
    for log_gamma_basepair in MIN_LOG_GAMMA_BASEPAIR..MAX_LOG_GAMMA_BASEPAIR + 1 {
      let basepair_count_posterior = (2. as Prob).powi(log_gamma_basepair) + 1.;
      let mut feature_scores = FeatureCountsPosterior::new(align_count_posterior);
      feature_scores.basepair_count_posterior = basepair_count_posterior;
      candidates.push((feature_scores, MeaStructAlign::new()));
    }
  }
  thread_pool.scoped(|scope| {
    let ref ref_2_align_prob_mats_with_rna_id_pairs = align_prob_mats_with_rna_id_pairs_fused;
    let ref ref_2_bpp_mats = bpp_mats_fused;
    let ref ref_2_insert_prob_set_pairs_with_rna_id_pairs = insert_prob_set_pairs_with_rna_id_pairs;
    for candidate in &mut candidates {
      scope.execute(move || {
        candidate.1 = consalign::<T, U>(
          fasta_records,
          ref_2_align_prob_mats_with_rna_id_pairs,
          ref_2_bpp_mats,
          &candidate.0,
          ref_2_insert_prob_set_pairs_with_rna_id_pairs,
        );
      });
    }
  });
  let mut sa = MeaStructAlign::new();
  let mut feature_scores = FeatureCountsPosterior::new(0.);
  for candidate in &candidates {
    let ref tmp_sa = candidate.1;
    if tmp_sa.sps > sa.sps {
      feature_scores = candidate.0.clone();
      sa = tmp_sa.clone();
    }
  }
  sa.sort();
  sa.struct_align.seq_align.seqs = fasta_records.iter().map(|x| x.seq.clone()).collect();
  let bpp_mat_alifold = if disable_alifold {
    SparseProbMat::<U>::default()
  } else {
    get_bpp_mat_alifold(&sa, &sa_file_path, fasta_records, output_dir_path)
  };
  let mix_bpp_mat = get_mix_bpp_mat(&sa, &bpp_mats_fused, &bpp_mat_alifold, disable_alifold);
  let sa_len = sa.struct_align.seq_align.pos_map_sets.len();
  let sa_len = U::from_usize(sa_len).unwrap();
  sa.struct_align.bp_pos_pairs = consalifold(&mix_bpp_mat, sa_len, BASEPAIR_COUNT_POSTERIOR_ALIFOLD);
  let output_file_path = output_dir_path.join(&format!("consalign.sth"));
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
    let ref seq_id = fasta_records[rna_id].fasta_id;
    buf_4_writer_2_output_file.push_str(seq_id);
    let mut stockholm_row = vec![' ' as Char; max_seq_id_len - seq_id.len() + 2];
    let ref fasta_record = fasta_records[rna_id];
    let ref seq = fasta_record.seq;
    let mut sa_row = (0..sa_len)
      .map(|x| {
        let pos_map = sa.struct_align.seq_align.pos_map_sets[x][rna_id].to_usize().unwrap();
        if pos_map == 0 {
          GAP
        } else {
          revert_char(seq[pos_map])
        }
      })
      .collect::<Vec<Char>>();
    stockholm_row.append(&mut sa_row);
    let stockholm_row = unsafe { from_utf8_unchecked(&stockholm_row) };
    buf_4_writer_2_output_file.push_str(&stockholm_row);
    buf_4_writer_2_output_file.push_str("\n");
  }
  buf_4_writer_2_output_file.push_str(descriptor);
  let mut stockholm_row = vec![' ' as Char; max_seq_id_len - descriptor_len + 2];
  let mut mea_css_str = get_mea_css_str(&sa, sa_len);
  stockholm_row.append(&mut mea_css_str);
  let stockholm_row = unsafe { from_utf8_unchecked(&stockholm_row) };
  buf_4_writer_2_output_file.push_str(&stockholm_row);
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
