extern crate consprob;
extern crate num_cpus;
extern crate petgraph;
extern crate rand;

pub mod trained_feature_score_sets;

pub use consprob::*;
pub use petgraph::{Graph, Directed, Outgoing};
pub use petgraph::graph::{DefaultIx, NodeIndex};
pub use rand::seq::SliceRandom;
pub use rand::Rng;

pub type Col = Vec<Base>;
pub type Cols = Vec<Col>;
pub type PosMaps<T> = Vec<T>;
pub type PosMapSets<T> = Vec<PosMaps<T>>;
pub type PosMapSetPair<T> = (PosMaps<T>, PosMaps<T>);
pub type PosMapSetPairs<T> = Vec<PosMapSetPair<T>>;
pub struct MeaSeqAlign<T> {
  pub cols: Cols,
  pub rightmost_bp_cols_with_cols: ColsWithCols<T>,
  pub right_bp_col_sets_with_cols: ColSetsWithCols<T>,
  pub pos_map_sets: PosMapSets<T>,
  pub rna_ids: RnaIds,
  pub ea: Mea,
  pub bp_col_pairs: SparsePosMat<T>,
}
pub struct ReadSeqAlign<T> {
  pub cols: Cols,
  pub bp_col_pairs: SparsePosMat<T>,
  pub pos_map_sets: PosMapSets<T>,
  pub fasta_records: FastaRecords,
}
pub type SparsePosMat<T> = HashSet<PosPair<T>>;
pub type RnaIds = Vec<RnaId>;
pub type MeaSeqAlignPair<'a, T> = (&'a MeaSeqAlign<T>, &'a MeaSeqAlign<T>);
pub type SparseMeaMat = HashMap<RnaIdPair, Mea>;
pub type ProgressiveTree = Graph<RnaId, Mea>;
pub type ClusterSizes = HashMap<RnaId, usize>;
pub type NodeIndexes = HashMap<RnaId, NodeIndex<DefaultIx>>;
pub type ColPairs<T> = HashSet<PosPair<T>>;
pub type ColsWithCols<T> = HashMap<T, T>;
pub type MeaMatsWithPosPairs<T> = HashMap<PosPair<T>, SparseProbMat<T>>;
pub type ColSetsWithCols<T> = HashMap<T, PosProbSeq<T>>;
pub type PosProbSeq<T> = Vec<(T, Prob)>;
#[derive(Clone)]
pub struct TrainDatumPosterior<T> {
  pub fasta_records: FastaRecords,
  pub observed_feature_count_sets: FeatureCountSetsPosterior,
  pub argmax_feature_count_sets: FeatureCountSetsPosterior,
  pub align_prob_mat_pairs_with_rna_id_pairs: AlignProbMatPairsWithRnaIdPairs<T>,
  pub prob_mat_sets: ProbMatSets<T>,
  pub max: FeatureCount,
  pub pos_map_sets: PosMapSets<T>,
  pub sa_len: usize,
  pub num_of_rnas: usize,
  pub bp_col_pairs: SparsePosMat<T>,
}
pub type TrainDataPosterior<T> = Vec<TrainDatumPosterior<T>>;
#[derive(Clone, Debug)]
pub struct FeatureCountSetsPosterior {
  pub basepair_count_posterior: FeatureCount,
  pub align_count_posterior: FeatureCount,
  pub basepair_count_offset: FeatureCount,
  pub align_count_offset: FeatureCount,
}

pub struct MeaCss<T> {
  pub bpa_pos_pairs: PosPairs<T>,
  pub ea: Mea,
}

impl<T> MeaCss<T> {
  pub fn new() -> MeaCss<T> {
    MeaCss {
      bpa_pos_pairs: PosPairs::<T>::new(),
      ea: 0.,
    }
  }
}

impl<T: Hash + Clone + Unsigned + PrimInt + FromPrimitive + Integer + Ord + Sync + Send + Display> TrainDatumPosterior<T> {
  pub fn origin() -> TrainDatumPosterior<T> {
    TrainDatumPosterior {
      fasta_records: FastaRecords::new(),
      observed_feature_count_sets: FeatureCountSetsPosterior::new(0.),
      argmax_feature_count_sets: FeatureCountSetsPosterior::new(NEG_INFINITY),
      align_prob_mat_pairs_with_rna_id_pairs: AlignProbMatPairsWithRnaIdPairs::<T>::default(),
      prob_mat_sets: ProbMatSets::<T>::default(),
      max: NEG_INFINITY,
      pos_map_sets: Vec::new(),
      sa_len: 0,
      num_of_rnas: 0,
      bp_col_pairs: SparsePosMat::<T>::default(),
    }
  }

  pub fn new(input_file_path: &Path, min_bpp: Prob, offset_4_max_gap_num: T, thread_pool: &mut Pool, mix_weight: Prob) -> TrainDatumPosterior<T> {
    let read_seq_align = read_sa_from_stockholm_file(input_file_path);
    let (prob_mat_sets, pct_align_prob_mat_pairs_with_rna_id_pairs) = consprob::<T>(
      thread_pool,
      &read_seq_align.fasta_records,
      min_bpp,
      offset_4_max_gap_num,
      false,
      true,
      mix_weight,
    );
    let sa_len = read_seq_align.cols.len();
    let num_of_rnas = read_seq_align.cols[0].len();
    TrainDatumPosterior {
      fasta_records: read_seq_align.fasta_records.clone(),
      observed_feature_count_sets: FeatureCountSetsPosterior::new(0.),
      argmax_feature_count_sets: FeatureCountSetsPosterior::new(NEG_INFINITY),
      align_prob_mat_pairs_with_rna_id_pairs: pct_align_prob_mat_pairs_with_rna_id_pairs,
      prob_mat_sets: prob_mat_sets,
      max: NEG_INFINITY,
      pos_map_sets: read_seq_align.pos_map_sets,
      sa_len: sa_len,
      num_of_rnas: num_of_rnas,
      bp_col_pairs: read_seq_align.bp_col_pairs,
    }
    /* let mut train_datum = TrainDatumPosterior {
      fasta_records: read_seq_align.fasta_records.clone(),
      observed_feature_count_sets: FeatureCountSetsPosterior::new(0.),
      argmax_feature_count_sets: FeatureCountSetsPosterior::new(NEG_INFINITY),
      align_prob_mat_pairs_with_rna_id_pairs: pct_align_prob_mat_pairs_with_rna_id_pairs,
      prob_mat_sets: prob_mat_sets,
      max: NEG_INFINITY,
    };
    train_datum.convert(&read_seq_align);
    train_datum */
  }

  // pub fn convert(&mut self, read_seq_align: &ReadSeqAlign<T>, feature_score_sets: &FeatureCountSetsPosterior) {
  pub fn update(&mut self, feature_score_sets: &FeatureCountSetsPosterior) {
    // for bp_col_pair in &read_seq_align.bp_col_pairs {
    for bp_col_pair in &self.bp_col_pairs {
      let (long_i, long_j) = (bp_col_pair.0.to_usize().unwrap(), bp_col_pair.1.to_usize().unwrap());
      // let ref pos_maps = read_seq_align.pos_map_sets[long_i];
      let ref pos_maps = self.pos_map_sets[long_i];
      // let ref pos_maps_2 = read_seq_align.pos_map_sets[long_j];
      let ref pos_maps_2 = self.pos_map_sets[long_j];
      let pos_map_pairs: Vec<(T, T)> = pos_maps.iter().zip(pos_maps_2.iter()).map(|(&x, &y)| (x, y)).collect();
      for (rna_id, pos_map_pair) in pos_map_pairs.iter().enumerate() {
        let ref bpp_mat = self.prob_mat_sets[rna_id].bpp_mat;
        match bpp_mat.get(pos_map_pair) {
          Some(&bpp) => {
            self.observed_feature_count_sets.basepair_count_posterior += bpp.ln() - feature_score_sets.basepair_count_offset;
            self.observed_feature_count_sets.basepair_count_offset -= feature_score_sets.basepair_count_posterior;
          }, None => {},
        }
      }
    }
    // let sa_len = read_seq_align.cols.len();
    // let num_of_rnas = read_seq_align.cols[0].len();
    let sa_len = self.sa_len;
    let num_of_rnas = self.num_of_rnas;
    for i in 0 .. sa_len {
      // let ref pos_maps = read_seq_align.pos_map_sets[i];
      let ref pos_maps = self.pos_map_sets[i];
      for j in 0 .. num_of_rnas {
        let pos = pos_maps[j];
        for k in j + 1 .. num_of_rnas {
          let pos_2 = pos_maps[k];
          let ref align_prob_mat = self.align_prob_mat_pairs_with_rna_id_pairs[&(j, k)].align_prob_mat;
          match align_prob_mat.get(&(pos, pos_2)) {
            Some(&align_prob) => {
              self.observed_feature_count_sets.align_count_posterior += align_prob.ln() - feature_score_sets.align_count_offset;
              self.observed_feature_count_sets.align_count_offset -= feature_score_sets.align_count_posterior;
            }, None => {},
          }
        }
      }
    }
  }
}

impl<T: Clone + Copy + Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display + Eq> MeaSeqAlign<T> {
  pub fn new() -> MeaSeqAlign<T> {
    MeaSeqAlign {
      cols: Cols::new(),
      rightmost_bp_cols_with_cols: ColsWithCols::<T>::default(),
      right_bp_col_sets_with_cols: ColSetsWithCols::<T>::default(),
      pos_map_sets: PosMapSets::<T>::new(),
      rna_ids: RnaIds::new(),
      ea: 0.,
      bp_col_pairs: SparsePosMat::<T>::default(),
    }
  }

  pub fn copy_subset(&mut self, mea_seq_align: &MeaSeqAlign<T>, indexes: &Vec<usize>, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob, feature_score_sets: &FeatureCountSetsPosterior) {
    let cols_len = mea_seq_align.cols.len();
    let num_of_rnas = mea_seq_align.cols[0].len();
    let col_gaps_only = vec![PSEUDO_BASE; num_of_rnas];
    let mut cols_subset = Cols::new();
    let mut pos_map_sets_subset = PosMapSets::new();
    for i in 0 .. cols_len {
      let ref col = mea_seq_align.cols[i];
      if *col == col_gaps_only {
        continue;
      }
      let ref pos_maps = mea_seq_align.pos_map_sets[i];
      let mut col_subset = Col::new();
      let mut pos_maps_subset = PosMaps::new();
      for (j, &c) in col.iter().enumerate() {
        let pos_map = pos_maps[j];
        if indexes.contains(&j) {
          col_subset.push(c);
          pos_maps_subset.push(pos_map);
        }
      }
      cols_subset.push(col_subset);
      pos_map_sets_subset.push(pos_maps_subset);
    }
    self.cols = cols_subset.clone();
    self.pos_map_sets = pos_map_sets_subset.clone();
    let mut rna_ids_subset = RnaIds::new();
    for (i, &r) in mea_seq_align.rna_ids.iter().enumerate() {
      if indexes.contains(&i) {
        rna_ids_subset.push(r);
      }
    }
    self.rna_ids = rna_ids_subset;
    self.set_right_bp_info(prob_mat_sets, min_bpp, feature_score_sets);
    self.ea = 0.;
  }

  pub fn set_right_bp_info(&mut self, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob, feature_score_sets: &FeatureCountSetsPosterior) {
    let sa_len = self.cols.len();
    for i in 0 .. sa_len {
      let ref pos_maps = self.pos_map_sets[i];
      for j in i + 1 .. sa_len {
        let ref pos_maps_2 = self.pos_map_sets[j];
        let pos_map_pairs: Vec<(T, T)> = pos_maps.iter().zip(pos_maps_2.iter()).map(|(&x, &y)| (x, y)).collect();
        let mut bpp_sum = 0.;
        let mut weight_sum = 0.;
        let mut count = 0;
        for (pos_map_pair, &rna_id) in pos_map_pairs.iter().zip(self.rna_ids.iter()) {
          let ref bpp_mat = prob_mat_sets[rna_id].bpp_mat;
          match bpp_mat.get(pos_map_pair) {
            Some(&bpp) => {
              bpp_sum += bpp;
              // weight_sum += feature_score_sets.basepair_count_posterior * bpp.ln() - feature_score_sets.basepair_count_offset;
              weight_sum += feature_score_sets.basepair_count_posterior * (bpp.ln() - feature_score_sets.basepair_count_offset);
              count += 1;
            }, None => {},
          }
        }
        let bpp_avg = if count > 0 {bpp_sum / count as Prob} else {0.};
        if bpp_avg >= min_bpp {
          let (i, j) = (T::from_usize(i).unwrap() + T::one(), T::from_usize(j).unwrap() + T::one());
          match self.right_bp_col_sets_with_cols.get_mut(&i) {
            Some(right_bp_cols) => {
              right_bp_cols.push((j, weight_sum));
            }, None => {
              let mut right_bp_cols = PosProbSeq::<T>::new();
              right_bp_cols.push((j, weight_sum));
              self.right_bp_col_sets_with_cols.insert(i, right_bp_cols);
            },
          }
        }
      }
      let i = T::from_usize(i).unwrap() + T::one();
      match self.right_bp_col_sets_with_cols.get(&i) {
        Some(right_bp_cols) => {
          let max = right_bp_cols.iter().map(|x| x.0).max().unwrap();
          self.rightmost_bp_cols_with_cols.insert(i, max);
        }, None => {},
      }
    }
  }
}

impl<T: Clone + Copy + Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display + Hash> ReadSeqAlign<T> {
  pub fn new() -> ReadSeqAlign<T> {
    ReadSeqAlign {
      cols: Cols::new(),
      bp_col_pairs: SparsePosMat::<T>::default(),
      pos_map_sets: PosMapSets::<T>::new(),
      fasta_records: FastaRecords::new(),
    }
  }
}

impl FeatureCountSetsPosterior {
  pub fn new(init_val: FeatureCount) -> FeatureCountSetsPosterior {
    FeatureCountSetsPosterior {
      basepair_count_posterior: init_val,
      align_count_posterior: init_val,
      basepair_count_offset: init_val,
      align_count_offset: init_val,
    }
  }

  pub fn get_len(&self) -> usize {
    4
  }

  pub fn update_regularizers(&self, regularizers: &mut Regularizers) {
    let mut regularizers_tmp = vec![0.; regularizers.len()];
    let mut offset = 0;
    let regularizer = get_regularizer(1, self.basepair_count_posterior * self.basepair_count_posterior);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.align_count_posterior * self.align_count_posterior);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.basepair_count_offset * self.basepair_count_offset);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let regularizer = get_regularizer(1, self.align_count_offset * self.align_count_offset);
    regularizers_tmp[offset] = regularizer;
    *regularizers = Array1::from(regularizers_tmp);
  }

  pub fn update<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&mut self, train_data: &[TrainDatumPosterior<T>], regularizers: &mut Regularizers)
  {
    let f = |_: &BfgsFeatureCounts| {
      self.get_cost(&train_data[..], regularizers) as BfgsFeatureCount
    };
    let g = |_: &BfgsFeatureCounts| {
      convert_feature_counts_2_bfgs_feature_counts(&self.get_grad(train_data, regularizers))
    };
    match bfgs(convert_feature_counts_2_bfgs_feature_counts(&convert_struct_2_vec_posterior(self)), f, g) {
      Ok(solution) => {
        *self = convert_vec_2_struct_posterior(&convert_bfgs_feature_counts_2_feature_counts(&solution));
      }, Err(_) => {
        println!("BFGS failed");
      },
    };
    self.update_regularizers(regularizers);
  }

  pub fn get_grad<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&self, train_data: &[TrainDatumPosterior<T>], regularizers: &Regularizers) -> FeatureCounts
  {
    let feature_scores = convert_struct_2_vec_posterior(self);
    let mut grad = FeatureCountSetsPosterior::new(0.);
    for train_datum in train_data {
      let ref obs = train_datum.observed_feature_count_sets;
      let ref expect = train_datum.argmax_feature_count_sets;
      let obs_count = obs.basepair_count_posterior;
      let expect_count = expect.basepair_count_posterior;
      grad.basepair_count_posterior -= obs_count - expect_count;
      let obs_count = obs.align_count_posterior;
      let expect_count = expect.align_count_posterior;
      grad.align_count_posterior -= obs_count - expect_count;
      let obs_count = obs.basepair_count_offset;
      let expect_count = expect.basepair_count_offset;
      grad.basepair_count_offset -= obs_count - expect_count;
      let obs_count = obs.align_count_offset;
      let expect_count = expect.align_count_offset;
      grad.align_count_offset -= obs_count - expect_count;
    }
    convert_struct_2_vec_posterior(&grad) + regularizers.clone() * feature_scores
  }

  pub fn get_cost<T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord>(&self, train_data: &[TrainDatumPosterior<T>], regularizers: &Regularizers) -> FeatureCount {
    let mut total_score = 0.;
    let feature_scores = convert_struct_2_vec_posterior(self);
    for train_datum in train_data {
      let ref obs = train_datum.observed_feature_count_sets;
      total_score += feature_scores.dot(&convert_struct_2_vec_posterior(obs));
      total_score -= train_datum.max;
    }
    let feature_scores = convert_struct_2_vec_posterior(self);
    let product = regularizers.clone() * feature_scores.clone();
    - total_score + product.dot(&feature_scores) / 2.
  }

  pub fn rand_init(&mut self) {
    let len = self.get_len();
    let std_deviation = 1. / (len as FeatureCount).sqrt();
    let normal = Normal::new(0., std_deviation).unwrap();
    let mut thread_rng = thread_rng();
    self.basepair_count_posterior = normal.sample(&mut thread_rng);
    self.align_count_posterior = normal.sample(&mut thread_rng);
    self.basepair_count_offset = normal.sample(&mut thread_rng);
    self.align_count_offset = normal.sample(&mut thread_rng);
  }
}

pub const GAP: Char = '-' as Char;
pub const MAX_ITER_REFINE: usize = 100;
pub const TRAINED_FEATURE_SCORE_SETS_FILE_PATH_POSTERIOR: &'static str = "../src/trained_feature_score_sets.rs";

pub fn consalign<T>(
  fasta_records: &FastaRecords,
  align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>,
  offset_4_max_gap_num: T,
  prob_mat_sets: &ProbMatSets<T>,
  min_bpp: Prob,
  feature_score_sets: &FeatureCountSetsPosterior,
) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  // let feature_score_sets = FeatureCountSetsPosterior::load_trained_score_params();
  let num_of_rnas = fasta_records.len();
  let mut mea_mat = SparseMeaMat::default();
  let mut progressive_tree = ProgressiveTree::new();
  let mut cluster_sizes = ClusterSizes::default();
  let mut node_indexes = NodeIndexes::default();
  for i in 0 .. num_of_rnas {
    let ref seq = fasta_records[i].seq;
    let ref sparse_bpp_mat = prob_mat_sets[i].bpp_mat;
    let converted_seq = convert_seq(seq, i, sparse_bpp_mat, &feature_score_sets);
    for j in i + 1 .. num_of_rnas {
      let ref seq_2 = fasta_records[j].seq;
      let ref sparse_bpp_mat_2 = prob_mat_sets[j].bpp_mat;
      let converted_seq_2 = convert_seq(seq_2, j, sparse_bpp_mat_2, &feature_score_sets);
      let pair_seq_align = get_mea_align(&(&converted_seq, &converted_seq_2), align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp, &feature_score_sets);
      mea_mat.insert((i, j), pair_seq_align.ea);
    }
    let node_index = progressive_tree.add_node(i);
    cluster_sizes.insert(i, 1);
    node_indexes.insert(i, node_index);
  }
  let mut new_cluster_id = num_of_rnas;
  while mea_mat.len() > 0 {
    let mut max = NEG_INFINITY;
    let mut argmax = (0, 0);
    for (cluster_id_pair, &ea) in &mea_mat {
      if ea > max {
        argmax = *cluster_id_pair;
        max = ea;
      }
    }
    let cluster_size_pair = (
      cluster_sizes.remove(&argmax.0).unwrap(),
      cluster_sizes.remove(&argmax.1).unwrap(),
    );
    let new_cluster_size = cluster_size_pair.0 + cluster_size_pair.1;
    cluster_sizes.insert(new_cluster_id, new_cluster_size);
    mea_mat.remove(&argmax);
    for &i in node_indexes.keys() {
      if i == argmax.0 || i == argmax.1 {
        continue;
      }
      let cluster_id_pair = if i < argmax.0 {(i, argmax.0)} else {(argmax.0, i)};
      let obtained_ea = mea_mat.remove(&cluster_id_pair).unwrap();
      let cluster_id_pair_2 = if i < argmax.1 {(i, argmax.1)} else {(argmax.1, i)};
      let obtained_ea_2 = mea_mat.remove(&cluster_id_pair_2).unwrap();
      let new_ea = (cluster_size_pair.0 as Mea * obtained_ea + cluster_size_pair.1 as Mea * obtained_ea_2) / new_cluster_size as Mea;
      mea_mat.insert((i, new_cluster_id), new_ea);
    }
    let new_node = progressive_tree.add_node(new_cluster_id);
    node_indexes.insert(new_cluster_id, new_node);
    let edge_len = max / 2.;
    let argmax_node_pair = (
      node_indexes.remove(&argmax.0).unwrap(),
      node_indexes.remove(&argmax.1).unwrap(),
    );
    progressive_tree.add_edge(new_node, argmax_node_pair.0, edge_len);
    progressive_tree.add_edge(new_node, argmax_node_pair.1, edge_len);
    new_cluster_id += 1;
  }
  let root = node_indexes[&(new_cluster_id - 1)];
  let mut mea_seq_align = recursive_mea_seq_align(&progressive_tree, root, align_prob_mat_pairs_with_rna_id_pairs, &fasta_records, offset_4_max_gap_num, prob_mat_sets, min_bpp, &feature_score_sets);
  for _ in 0 .. MAX_ITER_REFINE {
    iter_refine_seq_align(&mut mea_seq_align, align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp, &feature_score_sets);
  }
  for col in mea_seq_align.cols.iter_mut() {
    let mut pairs: Vec<(Base, RnaId)> = col.iter().zip(mea_seq_align.rna_ids.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by_key(|x| x.1.clone());
    *col = pairs.iter().map(|x| x.0).collect();
  }
  for pos_maps in mea_seq_align.pos_map_sets.iter_mut() {
    let mut pairs: Vec<(T, RnaId)> = pos_maps.iter().zip(mea_seq_align.rna_ids.iter()).map(|(&x, &y)| (x, y)).collect();
    pairs.sort_by_key(|x| x.1);
    *pos_maps = pairs.iter().map(|x| x.0).collect();
  }
  mea_seq_align.rna_ids.sort();
  mea_seq_align
}

pub fn iter_refine_seq_align<T>(mea_seq_align: &mut MeaSeqAlign<T>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, offset_4_max_gap_num: T, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob, feature_score_sets: &FeatureCountSetsPosterior)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut indexes: Vec<usize> = (0 .. mea_seq_align.rna_ids.len()).collect();
  let indexes_len = indexes.len();
  let mut rng = rand::thread_rng();
  indexes.shuffle(&mut rng);
  let split_at = rng.gen_range(1 .. indexes_len);
  let indexes_remain = indexes.split_off(split_at);
  let mut split_pair = (MeaSeqAlign::<T>::new(), MeaSeqAlign::<T>::new());
  split_pair.0.copy_subset(mea_seq_align, &indexes, prob_mat_sets, min_bpp, feature_score_sets);
  split_pair.1.copy_subset(mea_seq_align, &indexes_remain, prob_mat_sets, min_bpp, feature_score_sets);
  let tmp_align = get_mea_align(&(&split_pair.0, &split_pair.1), align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp, feature_score_sets);
  if tmp_align.ea > mea_seq_align.ea {
    *mea_seq_align = tmp_align;
  }
}

pub fn recursive_mea_seq_align<T>(progressive_tree: &ProgressiveTree, node: NodeIndex<DefaultIx>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, fasta_records: &FastaRecords, offset_4_max_gap_num: T, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob, feature_score_sets: &FeatureCountSetsPosterior) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let num_of_rnas = fasta_records.len();
  let rna_id = *progressive_tree.node_weight(node).unwrap();
  if rna_id < num_of_rnas {
    let ref seq = fasta_records[rna_id].seq;
    let ref sparse_bpp_mat = prob_mat_sets[rna_id].bpp_mat;
    convert_seq(seq, rna_id, sparse_bpp_mat, feature_score_sets)
  } else {
    let mut neighbors = progressive_tree.neighbors_directed(node, Outgoing).detach();
    let child = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_seq_align = recursive_mea_seq_align(progressive_tree, child, align_prob_mat_pairs_with_rna_id_pairs, fasta_records, offset_4_max_gap_num, prob_mat_sets, min_bpp, feature_score_sets);
    let child_2 = neighbors.next_node(progressive_tree).unwrap();
    let child_mea_seq_align_2 = recursive_mea_seq_align(progressive_tree, child_2, align_prob_mat_pairs_with_rna_id_pairs, fasta_records, offset_4_max_gap_num, prob_mat_sets, min_bpp, feature_score_sets);
    get_mea_align(&(&child_mea_seq_align, &child_mea_seq_align_2), align_prob_mat_pairs_with_rna_id_pairs, offset_4_max_gap_num, prob_mat_sets, min_bpp, feature_score_sets)
  }
}

pub fn convert_seq<T>(seq: &Seq, rna_id: RnaId, sparse_bpp_mat: &SparseProbMat<T>, feature_score_sets: &FeatureCountSetsPosterior) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut converted_seq = MeaSeqAlign::new();
  let seq_len = seq.len();
  converted_seq.cols = seq[1 .. seq_len - 1].iter().map(|&x| vec![x]).collect();
  converted_seq.pos_map_sets = (1 .. seq.len() - 1).map(|x| vec![T::from_usize(x).unwrap()]).collect();
  converted_seq.rna_ids = vec![rna_id];
  for (pos_pair, &bpp) in sparse_bpp_mat {
    let (i, j) = *pos_pair;
    // let weight = feature_score_sets.basepair_count_posterior * bpp.ln() - feature_score_sets.basepair_count_offset;
    let weight = feature_score_sets.basepair_count_posterior * (bpp.ln() - feature_score_sets.basepair_count_offset);
    match converted_seq.right_bp_col_sets_with_cols.get_mut(&i) {
      Some(right_bp_cols) => {
        right_bp_cols.push((j, weight));
      }, None => {
        let mut right_bp_cols = PosProbSeq::<T>::new();
        right_bp_cols.push((j, weight));
        converted_seq.right_bp_col_sets_with_cols.insert(i, right_bp_cols);
      },
    }
    for (&i, right_bp_cols) in &converted_seq.right_bp_col_sets_with_cols {
      let max = right_bp_cols.iter().map(|x| x.0).max().unwrap();
      converted_seq.rightmost_bp_cols_with_cols.insert(i, max);
    }
  }
  converted_seq
}

pub fn get_mea_align<'a, T>(seq_align_pair: &MeaSeqAlignPair<'a, T>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, offset_4_max_gap_num: T, prob_mat_sets: &ProbMatSets<T>, min_bpp: Prob, feature_score_sets: &FeatureCountSetsPosterior) -> MeaSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let seq_align_len_pair = (seq_align_pair.0.cols.len(), seq_align_pair.1.cols.len());
  let max_gap_num = offset_4_max_gap_num
    + T::from_usize(max(seq_align_len_pair.0, seq_align_len_pair.1) - min(seq_align_len_pair.0, seq_align_len_pair.1))
      .unwrap();
  let max_gap_num_4_il = max(
    min(max_gap_num, T::from_usize(MAX_GAP_NUM_4_IL).unwrap()),
    T::from_usize(MIN_GAP_NUM_4_IL).unwrap(),
  );
  let mut align_prob_mat_avg = SparseProbMat::<T>::default();
  let ref rna_ids = seq_align_pair.0.rna_ids;
  let ref rna_ids_2 = seq_align_pair.1.rna_ids;
  let ref pos_map_sets = seq_align_pair.0.pos_map_sets;
  let ref pos_map_sets_2 = seq_align_pair.1.pos_map_sets;
  let seq_align_len_pair = (
    T::from_usize(seq_align_len_pair.0).unwrap(),
    T::from_usize(seq_align_len_pair.1).unwrap(),
  );
  let pseudo_col_quadruple = (
    T::zero(),
    seq_align_len_pair.0 + T::one(),
    T::zero(),
    seq_align_len_pair.1 + T::one(),
  );
  let max = max(max_gap_num, max_gap_num_4_il);
  for i in range_inclusive(T::one(), seq_align_len_pair.0) {
    let long_i = i.to_usize().unwrap();
    let ref pos_maps = pos_map_sets[long_i - 1];
    for j in range_inclusive(T::one(), seq_align_len_pair.1) {
      let col_pair = (i, j);
      if !is_min_gap_ok(&col_pair, &pseudo_col_quadruple, max) {continue;}
      let long_j = j.to_usize().unwrap();
      let ref pos_maps_2 = pos_map_sets_2[long_j - 1];
      let mut align_prob_sum = 0.;
      // let mut count = 0;
      for (&rna_id, &pos) in rna_ids.iter().zip(pos_maps.iter()) {
        for (&rna_id_2, &pos_2) in rna_ids_2.iter().zip(pos_maps_2.iter()) {
          let ordered_rna_id_pair = if rna_id < rna_id_2 {(rna_id, rna_id_2)} else {(rna_id_2, rna_id)};
          let ref align_prob_mat = align_prob_mat_pairs_with_rna_id_pairs[&ordered_rna_id_pair].align_prob_mat;
          let pos_pair = if rna_id < rna_id_2 {(pos, pos_2)} else {(pos_2, pos)};
          match align_prob_mat.get(&pos_pair) {
            Some(&align_prob) => {
              // align_prob_sum += feature_score_sets.align_count_posterior * align_prob.ln() - feature_score_sets.align_count_offset;
              align_prob_sum += feature_score_sets.align_count_posterior * (align_prob.ln() - feature_score_sets.align_count_offset);
              // count += 1;
            }, None => {},
          }
        }
      }
      // let align_prob_avg = if count > 0 {align_prob_sum / count as Prob} else {0.};
      // let align_prob_avg = align_prob_sum;
      // align_prob_mat_avg.insert(col_pair, align_prob_avg);
      align_prob_mat_avg.insert(col_pair, align_prob_sum);
    }
  }
  let mut mea_mats_with_col_pairs = MeaMatsWithPosPairs::default();
  for i in range_inclusive(T::one(), seq_align_len_pair.0).rev() {
    match seq_align_pair.0.rightmost_bp_cols_with_cols.get(&i) {
      Some(&j) => {
        for k in range_inclusive(T::one(), seq_align_len_pair.1).rev() {
          let col_pair_left = (i, k);
          if !is_min_gap_ok(&col_pair_left, &pseudo_col_quadruple, max_gap_num_4_il) {continue;}
          match seq_align_pair.1.rightmost_bp_cols_with_cols.get(&k) {
            Some(&l) => {
              let col_quadruple = (i, j, k, l);
              let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg, &col_quadruple, true);
              update_mea_mats_with_col_pairs(&mut mea_mats_with_col_pairs, &col_pair_left, seq_align_pair, &mea_mat, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg);
            }, None => {},
          }
        }
      }, None => {},
    }
  }
  let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg, &pseudo_col_quadruple, false);
  let mut new_mea_seq_align = MeaSeqAlign::new();
  new_mea_seq_align.ea = mea_mat[&(pseudo_col_quadruple.1 - T::one(), pseudo_col_quadruple.3 - T::one())];
  // let rna_num_pair = (seq_align_pair.0.rna_ids.len(), seq_align_pair.1.rna_ids.len());
  let mut new_rna_ids = seq_align_pair.0.rna_ids.clone();
  let mut rna_ids_append = seq_align_pair.1.rna_ids.clone();
  new_rna_ids.append(&mut rna_ids_append);
  new_mea_seq_align.rna_ids = new_rna_ids;
  let mut bp_pos_map_set_pairs = PosMapSetPairs::<T>::new();
  traceback(&mut new_mea_seq_align, seq_align_pair, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &pseudo_col_quadruple, &mea_mats_with_col_pairs, 0, &align_prob_mat_avg, &mut bp_pos_map_set_pairs);
  let sa_len = new_mea_seq_align.cols.len();
  for bp_pos_map_set_pair in &bp_pos_map_set_pairs {
    for i in 0 .. sa_len {
      let ref pos_maps = new_mea_seq_align.pos_map_sets[i];
      if *pos_maps != bp_pos_map_set_pair.0 {
        continue;
      }
      let short_i = T::from_usize(i).unwrap();
      for j in i + 1 .. sa_len {
        let ref pos_maps_2 = new_mea_seq_align.pos_map_sets[j];
        if *pos_maps_2 == bp_pos_map_set_pair.1 {
          let short_j = T::from_usize(j).unwrap();
          new_mea_seq_align.bp_col_pairs.insert((short_i, short_j));
          break;
        }
      }
    }
  }
  new_mea_seq_align.set_right_bp_info(prob_mat_sets, min_bpp, feature_score_sets);
  new_mea_seq_align
}

pub fn traceback <'a, T>(new_mea_seq_align: &mut MeaSeqAlign<T>, seq_align_pair: &MeaSeqAlignPair<'a, T>, max_gap_num: T, max_gap_num_4_il: T, pseudo_col_quadruple: &PosQuadruple<T>, col_quadruple: &PosQuadruple<T>, mea_mats_with_col_pairs: &MeaMatsWithPosPairs<T>, offset: usize, align_prob_mat_avg: &SparseProbMat<T>, bp_pos_map_set_pairs: &mut PosMapSetPairs<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let rna_num_pair = (seq_align_pair.0.rna_ids.len(), seq_align_pair.1.rna_ids.len());
  let mut mea;
  let mea_mat = get_mea_mat(&mea_mats_with_col_pairs, max_gap_num, max_gap_num_4_il, &pseudo_col_quadruple, &align_prob_mat_avg, &col_quadruple, col_quadruple != pseudo_col_quadruple);
  let (i, j, k, l) = *col_quadruple;
  let (mut u, mut v) = (j - T::one(), l - T::one());
  while u > i || v > k {
    let col_pair = (u, v);
    mea = mea_mat[&col_pair];
    let (long_u, long_v) = (u.to_usize().unwrap(), v.to_usize().unwrap());
    if u > i && v > k {
      let align_prob_avg = align_prob_mat_avg[&col_pair];
      let col_pair_4_match = (u - T::one(), v - T::one());
      let ea = mea_mat[&col_pair_4_match] + align_prob_avg;
      if ea == mea {
        let mut new_col = seq_align_pair.0.cols[long_u - 1].clone();
        let mut col_append = seq_align_pair.1.cols[long_v - 1].clone();
        new_col.append(&mut col_append);
        new_mea_seq_align.cols.insert(offset, new_col);
        let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[long_u - 1].clone();
        let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_v - 1].clone();
        new_pos_map_sets.append(&mut pos_map_sets_append);
        new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
        u = u - T::one();
        v = v - T::one();
        continue;
      }
      let mut is_basepair_match_found = false;
      match mea_mats_with_col_pairs.get(&col_pair) {
        Some(mea_mat_4_bpas) => {
          for (col_pair_left, mea_4_bpa) in mea_mat_4_bpas {
            if !(i < col_pair_left.0 && k < col_pair_left.1) {continue;}
            let col_pair_4_match = (col_pair_left.0 - T::one(), col_pair_left.1 - T::one());
            match mea_mat.get(&col_pair_4_match) {
              Some(&ea) => {
                let ea = ea + mea_4_bpa;
                if ea == mea {
                  let mut new_col = seq_align_pair.0.cols[long_u - 1].clone();
                  let mut col_append = seq_align_pair.1.cols[long_v - 1].clone();
                  new_col.append(&mut col_append);
                  new_mea_seq_align.cols.insert(offset, new_col);
                  let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[long_u - 1].clone();
                  let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_v - 1].clone();
                  new_pos_map_sets.append(&mut pos_map_sets_append);
                  new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets.clone());
                  let long_col_pair_left = (col_pair_left.0.to_usize().unwrap(), col_pair_left.1.to_usize().unwrap());
                  let mut new_col = seq_align_pair.0.cols[long_col_pair_left.0 - 1].clone();
                  let mut col_append = seq_align_pair.1.cols[long_col_pair_left.1 - 1].clone();
                  new_col.append(&mut col_append);
                  new_mea_seq_align.cols.insert(offset, new_col);
                  let mut new_pos_map_sets_2 = seq_align_pair.0.pos_map_sets[long_col_pair_left.0 - 1].clone();
                  let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_col_pair_left.1 - 1].clone();
                  new_pos_map_sets_2.append(&mut pos_map_sets_append);
                  new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets_2.clone());
                  bp_pos_map_set_pairs.push((new_pos_map_sets_2, new_pos_map_sets));
                  traceback(new_mea_seq_align, seq_align_pair, max_gap_num, max_gap_num_4_il, pseudo_col_quadruple, &(col_pair_left.0, u, col_pair_left.1, v), mea_mats_with_col_pairs, offset + 1, align_prob_mat_avg, bp_pos_map_set_pairs);
                  u = col_pair_4_match.0;
                  v = col_pair_4_match.1;
                  is_basepair_match_found = true;
                  break;
                }
              }, None => {},
            }
          }
        }, None => {},
      }
      if is_basepair_match_found {
        continue;
      }
    }
    if u > i {
      match mea_mat.get(&(u - T::one(), v)) {
        Some(&ea) => {
          if ea == mea {
            let mut new_col = seq_align_pair.0.cols[long_u - 1].clone();
            let mut col_append = vec![PSEUDO_BASE; rna_num_pair.1];
            new_col.append(&mut col_append);
            new_mea_seq_align.cols.insert(offset, new_col);
            let mut new_pos_map_sets = seq_align_pair.0.pos_map_sets[long_u - 1].clone();
            let mut pos_map_sets_append = vec![T::zero(); rna_num_pair.1];
            new_pos_map_sets.append(&mut pos_map_sets_append);
            new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
            u = u - T::one();
            continue;
          }
        }, None => {},
      }
    }
    if v > k {
      match mea_mat.get(&(u, v - T::one())) {
        Some(&ea) => {
          if ea == mea {
            let mut new_col = vec![PSEUDO_BASE; rna_num_pair.0];
            let mut col_append = seq_align_pair.1.cols[long_v - 1].clone();
            new_col.append(&mut col_append);
            new_mea_seq_align.cols.insert(offset, new_col);
            let mut new_pos_map_sets = vec![T::zero(); rna_num_pair.0];
            let mut pos_map_sets_append = seq_align_pair.1.pos_map_sets[long_v - 1].clone();
            new_pos_map_sets.append(&mut pos_map_sets_append);
            new_mea_seq_align.pos_map_sets.insert(offset, new_pos_map_sets);
            v = v - T::one();
          }
        }, None => {},
      }
    }
  }
}

pub fn get_mea_mat<T>(mea_mats_with_col_pairs: &MeaMatsWithPosPairs<T>, max_gap_num: T, max_gap_num_4_il: T, pseudo_col_quadruple: &PosQuadruple<T>, align_prob_mat_avg: &SparseProbMat<T>, col_quadruple: &PosQuadruple<T>, is_internal: bool) -> SparseProbMat<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let (i, j, k, l) = *col_quadruple;
  let mut mea_mat = SparseProbMat::<T>::default();
  for u in range(i, j) {
    for v in range(k, l) {
      let col_pair = (u, v);
      if !is_min_gap_ok(&col_pair, pseudo_col_quadruple, if is_internal {max_gap_num_4_il} else {max_gap_num}) {continue;}
      if u == i && v == k {
        mea_mat.insert(col_pair, 0.);
        continue;
      }
      let mut mea = NEG_INFINITY;
      match mea_mats_with_col_pairs.get(&col_pair) {
        Some(mea_mat_4_bpas) => {
          for (col_pair_left, mea_4_bpa) in mea_mat_4_bpas {
            let col_pair_4_match = (col_pair_left.0 - T::one(), col_pair_left.1 - T::one());
            if !(i < col_pair_left.0 && k < col_pair_left.1) {continue;}
            match mea_mat.get(&col_pair_4_match) {
              Some(&ea) => {
                let ea = ea + mea_4_bpa;
                if ea > mea {
                  mea = ea;
                }
              }, None => {},
            }
          }
        }, None => {},
      }
      let col_pair_4_match = (u - T::one(), v - T::one());
      match mea_mat.get(&col_pair_4_match) {
        Some(&ea) => {
          let align_prob_avg = align_prob_mat_avg[&col_pair];
          let ea = ea + align_prob_avg;
          if ea > mea {
            mea = ea;
          }
        }, None => {},
      }
      let col_pair_4_insert = (u - T::one(), v);
      match mea_mat.get(&col_pair_4_insert) {
        Some(&ea) => {
          if ea > mea {
            mea = ea;
          }
        }, None => {},
      }
      let col_pair_4_insert_2 = (u, v - T::one());
      match mea_mat.get(&col_pair_4_insert_2) {
        Some(&ea) => {
          if ea > mea {
            mea = ea;
          }
        }, None => {},
      }
      mea_mat.insert(col_pair, mea);
    }
  }
  mea_mat
}

pub fn update_mea_mats_with_col_pairs<'a, T>(mea_mats_with_col_pairs: &mut MeaMatsWithPosPairs<T>, col_pair_left: &PosPair<T>, seq_align_pair: &MeaSeqAlignPair<'a, T>, mea_mat: &SparseProbMat<T>, max_gap_num_4_il: T, pseudo_col_quadruple: &PosQuadruple<T>, align_prob_mat_avg: &SparseProbMat<T>)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let (i, k) = *col_pair_left;
  // let (long_i, long_k) = (i.to_usize().unwrap(), k.to_usize().unwrap());
  // let ref rna_ids = seq_align_pair.0.rna_ids;
  // let ref rna_ids_2 = seq_align_pair.1.rna_ids;
  // let ref pos_map_sets = seq_align_pair.0.pos_map_sets;
  // let ref pos_map_sets_2 = seq_align_pair.1.pos_map_sets;
  let ref right_bp_cols = seq_align_pair.0.right_bp_col_sets_with_cols[&i];
  let ref right_bp_cols_2 = seq_align_pair.1.right_bp_col_sets_with_cols[&k];
  let align_prob_avg_left = align_prob_mat_avg[&col_pair_left];
  for &(j, bpp_avg) in right_bp_cols.iter() {
    for &(l, bpp_avg_2) in right_bp_cols_2.iter() {
      let col_pair_right = (j, l);
      if !is_min_gap_ok(&col_pair_right, &pseudo_col_quadruple, max_gap_num_4_il) {continue;}
      let align_prob_avg_right = align_prob_mat_avg[&col_pair_right];
      let basepair_align_prob_avg = bpp_avg + bpp_avg_2 + align_prob_avg_left + align_prob_avg_right;
      let mea_4_bpa = basepair_align_prob_avg + mea_mat[&(j - T::one(), l - T::one())];
      match mea_mats_with_col_pairs.get_mut(&col_pair_right) {
        Some(mea_mat_4_bpas) => {
          mea_mat_4_bpas.insert(*col_pair_left, mea_4_bpa);
        }, None => {
          let mut mea_mat_4_bpas = SparseProbMat::default();
          mea_mat_4_bpas.insert(*col_pair_left, mea_4_bpa);
          mea_mats_with_col_pairs.insert(col_pair_right, mea_mat_4_bpas);
        },
      }
    }
  }
}

pub fn revert_char(c: Base) -> u8 {
  match c {
    A => BIG_A,
    C => BIG_C,
    G => BIG_G,
    U => BIG_U,
    PSEUDO_BASE => {GAP},
    _ => {assert!(false); GAP},
  }
}

pub fn consalifold<T>(mix_bpp_mat: &ProbMat, gamma: Prob, sa: &MeaSeqAlign<T>) -> MeaCss<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer,
{
  let sa_len = sa.cols.len();
  let mut mea_mat = vec![vec![0.; sa_len]; sa_len];
  let sa_len = T::from_usize(sa_len).unwrap();
  let gamma_plus_1 = gamma + 1.;
  for sub_sa_len in range_inclusive(T::one(), sa_len) {
    for i in range_inclusive(T::zero(), sa_len - sub_sa_len) {
      let j = i + sub_sa_len - T::one();
      let (long_i, long_j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
      if i == j {
        continue;
      }
      let mut mea = mea_mat[long_i + 1][long_j];
      let ea = mea_mat[long_i][long_j - 1];
      if ea > mea {
        mea = ea;
      }
      let ea = mea_mat[long_i + 1][long_j - 1] + gamma_plus_1 * mix_bpp_mat[long_i][long_j] - 1.;
      if ea > mea {
        mea = ea;
      }
      for k in long_i .. long_j {
        let ea = mea_mat[long_i][k] + mea_mat[k + 1][long_j];
        if ea > mea {
          mea = ea;
        }
      }
      mea_mat[long_i][long_j] = mea;
    }
  }
  let mut mea_css = MeaCss::new();
  let mut pos_pair_stack = vec![(T::zero(), sa_len - T::one())];
  while pos_pair_stack.len() > 0 {
    let pos_pair = pos_pair_stack.pop().expect("Failed to pop an element of a vector.");
    let (i, j) = pos_pair;
    if j <= i {continue;}
    let (long_i, long_j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
    let mea = mea_mat[long_i][long_j];
    if mea == mea_mat[long_i + 1][long_j] {
      pos_pair_stack.push((i + T::one(), j));
    } else if mea == mea_mat[long_i][long_j - 1] {
      pos_pair_stack.push((i, j - T::one()));
    } else if mea == mea_mat[long_i + 1][long_j - 1] + gamma_plus_1 * mix_bpp_mat[long_i][long_j] - 1. {
      pos_pair_stack.push((i + T::one(), j - T::one()));
      mea_css.bpa_pos_pairs.push(pos_pair);
    } else {
      for k in range(i, j) {
        let long_k = k.to_usize().unwrap();
        if mea == mea_mat[long_i][long_k] + mea_mat[long_k + 1][long_j] {
          pos_pair_stack.push((i, k));
          pos_pair_stack.push((k + T::one(), j));
          break;
        }
      }
    }
  }
  mea_css.ea = mea_mat[0][sa_len.to_usize().unwrap() - 1];
  mea_css
}

pub fn constrain_posterior<'a, T>(
  thread_pool: &mut Pool,
  train_data: &mut TrainDataPosterior<T>,
  offset_4_max_gap_num: T,
  output_file_path: &Path,
  min_bpp: Prob,
)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut feature_score_sets = FeatureCountSetsPosterior::new(0.);
  feature_score_sets.rand_init();
  for train_datum in train_data.iter_mut() {
    train_datum.update(&feature_score_sets);
  }
  let mut old_feature_score_sets = feature_score_sets.clone();
  let mut old_cost = INFINITY;
  let mut costs = Probs::new();
  let mut count = 0;
  let mut regularizers = Regularizers::from(vec![1.; feature_score_sets.get_len()]);
  let num_of_data = train_data.len() as FeatureCount;
  loop {
    let ref ref_2_feature_score_sets = feature_score_sets;
    thread_pool.scoped(|scope| {
      for train_datum in train_data.iter_mut() {
        train_datum.argmax_feature_count_sets = FeatureCountSetsPosterior::new(0.);
        let ref fasta_records = train_datum.fasta_records;
        let ref align_prob_mat_pairs_with_rna_id_pairs = train_datum.align_prob_mat_pairs_with_rna_id_pairs;
        let ref prob_mat_sets = train_datum.prob_mat_sets;
        let ref mut argmax_feature_count_sets = train_datum.argmax_feature_count_sets;
        scope.execute(move || {
          let argmax = consalign::<T>(
            fasta_records,
            align_prob_mat_pairs_with_rna_id_pairs,
            offset_4_max_gap_num,
            prob_mat_sets,
            min_bpp,
            ref_2_feature_score_sets,
          );
          convert_seq_align_2_param_counts(argmax_feature_count_sets, &argmax, prob_mat_sets, align_prob_mat_pairs_with_rna_id_pairs, ref_2_feature_score_sets);
        });
      }
    });
    feature_score_sets.update(&train_data, &mut regularizers);
    for train_datum in train_data.iter_mut() {
      train_datum.update(&feature_score_sets);
    }
    let cost = feature_score_sets.get_cost(&train_data[..], &regularizers);
    if old_cost.is_finite() && (old_cost - cost) / num_of_data <= LEARNING_TOLERANCE {
      feature_score_sets = old_feature_score_sets.clone();
      break;
    }
    costs.push(cost);
    old_feature_score_sets = feature_score_sets.clone();
    old_cost = cost;
    println!("Epoch {} finished (current cost = {})", count + 1, cost);
    count += 1;
  }
  write_feature_score_sets_trained_posterior(&feature_score_sets);
  write_costs(&costs, output_file_path);
}

pub fn convert_seq_align_2_param_counts<T>(feature_count_sets: &mut FeatureCountSetsPosterior, argmax: &MeaSeqAlign<T>, prob_mat_sets: &ProbMatSets<T>, align_prob_mat_pairs_with_rna_id_pairs: &AlignProbMatPairsWithRnaIdPairs<T>, feature_score_sets: &FeatureCountSetsPosterior)
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  for bp_col_pair in &argmax.bp_col_pairs {
    let (long_i, long_j) = (bp_col_pair.0.to_usize().unwrap(), bp_col_pair.1.to_usize().unwrap());
    let ref pos_maps = argmax.pos_map_sets[long_i];
    let ref pos_maps_2 = argmax.pos_map_sets[long_j];
    let pos_map_pairs: Vec<(T, T)> = pos_maps.iter().zip(pos_maps_2.iter()).map(|(&x, &y)| (x, y)).collect();
    for (pos_map_pair, &rna_id) in pos_map_pairs.iter().zip(argmax.rna_ids.iter()) {
      let ref bpp_mat = prob_mat_sets[rna_id].bpp_mat;
      match bpp_mat.get(pos_map_pair) {
        Some(&bpp) => {
          feature_count_sets.basepair_count_posterior += bpp.ln() - feature_score_sets.basepair_count_offset;
          feature_count_sets.basepair_count_offset -= feature_score_sets.basepair_count_posterior;
        }, None => {},
      }
    }
  }
  let sa_len = argmax.cols.len();
  let num_of_rnas = argmax.cols[0].len();
  for i in 0 .. sa_len {
    let ref pos_maps = argmax.pos_map_sets[i];
    for j in 0 .. num_of_rnas {
      let pos = pos_maps[j];
      for k in j + 1 .. num_of_rnas {
        let pos_2 = pos_maps[k];
        let ref align_prob_mat = align_prob_mat_pairs_with_rna_id_pairs[&(j, k)].align_prob_mat;
        match align_prob_mat.get(&(pos, pos_2)) {
          Some(&align_prob) => {
            feature_count_sets.align_count_posterior += align_prob.ln() - feature_score_sets.align_count_offset;
            feature_count_sets.align_count_offset -= feature_score_sets.align_count_posterior;
          }, None => {},
        }
      }
    }
  }
}

pub fn read_sa_from_stockholm_file<T>(stockholm_file_path: &Path) -> ReadSeqAlign<T>
where
  T: Unsigned + PrimInt + Hash + FromPrimitive + Integer + Ord + Sync + Send + Display,
{
  let mut sa = ReadSeqAlign::<T>::new();
  // let mut fasta_records = FastaRecords::new();
  let reader_2_stockholm_file = BufReader::new(File::open(stockholm_file_path).unwrap());
  let mut seqs = Vec::<Seq>::new();
  let mut bp_col_pairs = SparsePosMat::<T>::default();
  let mut seq_ids = SeqIds::new();
  for string in reader_2_stockholm_file.lines() {
    let string = string.unwrap();
    if string.starts_with("#=GC SS_cons") {
      let substrings: Vec<&str> = string.split_whitespace().collect();
      let ref css = substrings[2];
      let mut stack = Vec::new();
      for (i, c) in css.chars().enumerate() {
        if c == '(' {
          stack.push(i);
        } else if c == ')' {
          let i = T::from_usize(i).unwrap();
          let pos = stack.pop().unwrap();
          let pos = T::from_usize(pos).unwrap();
          bp_col_pairs.insert((pos, i));
        }
      }
    }
    if string.len() == 0 || string.starts_with("#") {
      continue;
    } else if string.starts_with("//") {
      break;
    }
    let substrings: Vec<&str> = string.split_whitespace().collect();
    let seq_id = substrings[0];
    seq_ids.push(SeqId::from(seq_id));
    let seq = substrings[1];
    let seq = seq.chars().map(|x| convert_sa_char(x as u8)).collect();
    seqs.push(seq);
  }
  sa.bp_col_pairs = bp_col_pairs;
  let align_len = seqs[0].len();
  for i in 0 .. align_len {
    let col = seqs.iter().map(|x| x[i]).collect();
    sa.cols.push(col);
  }
  let num_of_rnas = sa.cols[0].len();
  let mut seq_lens = vec![0 as usize; num_of_rnas];
  sa.pos_map_sets = vec![vec![T::zero(); num_of_rnas]; align_len];
  let mut fasta_records = vec![FastaRecord::origin(); num_of_rnas];
  for i in 0 .. align_len {
    for j in 0 .. num_of_rnas {
      let base = sa.cols[i][j];
      if base != PSEUDO_BASE {
        fasta_records[j].seq.push(base);
        seq_lens[j] += 1;
        sa.pos_map_sets[i][j] = T::from_usize(seq_lens[j]).unwrap();
      }
    }
  }
  for i in 0 .. num_of_rnas {
    fasta_records[i].seq.insert(0, PSEUDO_BASE);
    fasta_records[i].seq.push(PSEUDO_BASE);
    fasta_records[i].fasta_id = seq_ids[i].clone();
  }
  sa.fasta_records = fasta_records;
  sa 
}

pub fn convert_vec_2_struct_posterior(feature_counts: &FeatureCounts) -> FeatureCountSetsPosterior {
  let mut f = FeatureCountSetsPosterior::new(0.);
  let mut offset = 0;
  f.basepair_count_posterior = feature_counts[offset];
  offset += 1;
  f.align_count_posterior = feature_counts[offset];
  offset += 1;
  f.basepair_count_offset = feature_counts[offset];
  offset += 1;
  f.align_count_posterior = feature_counts[offset];
  f
}

pub fn convert_struct_2_vec_posterior(feature_count_sets: &FeatureCountSetsPosterior) -> FeatureCounts {
  let f = feature_count_sets;
  let mut feature_counts = vec![0.; f.get_len()];
  let mut offset = 0;
  feature_counts[offset] = f.basepair_count_posterior;
  offset += 1;
  feature_counts[offset] = f.align_count_posterior;
  offset += 1;
  feature_counts[offset] = f.basepair_count_offset;
  offset += 1;
  feature_counts[offset] = f.align_count_offset;
  Array::from(feature_counts)
}

pub fn write_feature_score_sets_trained_posterior(feature_score_sets: &FeatureCountSetsPosterior) {
  let mut writer_2_trained_feature_score_sets_file = BufWriter::new(File::create(TRAINED_FEATURE_SCORE_SETS_FILE_PATH_POSTERIOR).unwrap());
  let buf_4_writer_2_trained_feature_score_sets_file = format!("use FeatureCountSetsPosterior;\nimpl FeatureCountSetsPosterior {{\npub fn load_trained_score_params() -> FeatureCountSetsPosterior {{\nFeatureCountSetsPosterior {{\nbasepair_count_posterior: {}, align_count_posterior: {}, basepair_count_offset: {}, align_count_offset: {}\n}}\n}}\n}}", feature_score_sets.basepair_count_posterior, feature_score_sets.align_count_posterior, feature_score_sets.basepair_count_offset, feature_score_sets.align_count_offset);
  let _ = writer_2_trained_feature_score_sets_file.write_all(buf_4_writer_2_trained_feature_score_sets_file.as_bytes());
}
