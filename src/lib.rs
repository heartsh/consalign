extern crate consprob_trained;
extern crate petgraph;

pub use consprob_trained::*;
pub use petgraph::graph::{DefaultIx, NodeIndex};
pub use petgraph::{Directed, Graph, Outgoing};
pub use std::env;
pub use std::fs::create_dir;
pub use std::fs::remove_file;
pub use std::fs::File;
pub use std::io::{BufRead, BufWriter};
pub use std::process::{Command, Output};

pub type PosMapsPair<T> = (PosMaps<T>, PosMaps<T>);
pub type PosMapsPairs<T> = Vec<PosMapsPair<T>>;
#[derive(Clone)]
pub struct AlignfoldWrapped<T, U> {
  pub alignfold: Alignfold<T, U>,
  pub rightmost_basepairs_hashed_cols: ColsHashedCols<U>,
  pub right_basepairs_hashed_cols: ColSetsHashedCols<U>,
  pub rna_ids: RnaIds,
  pub accuracy: Score,
}
pub type RnaIds = Vec<RnaId>;
pub type AlignfoldPair<'a, T, U> = (&'a AlignfoldWrapped<T, U>, &'a AlignfoldWrapped<T, U>);
pub type GuideTreeScores = HashMap<RnaIdPair, Score>;
pub type GuideTree = Graph<RnaId, Score>;
pub type ClusterSizes = HashMap<RnaId, usize>;
pub type NodeIndexes = HashMap<RnaId, NodeIndex<DefaultIx>>;
pub type ColPairs<T> = HashSet<PosPair<T>>;
pub type ColsHashedCols<T> = HashMap<T, T>;
pub type ScoreMatsHashedPoss<T> = HashMap<PosPair<T>, SparseScoreMat<T>>;
pub type ColSetsHashedCols<T> = HashMap<T, ColsScored<T>>;
pub type ColsScored<T> = Vec<(T, Score)>;
#[derive(Clone, Debug)]
pub struct AlignfoldHyperparams {
  pub param_basepair: Score,
  pub param_match: Score,
}
pub type SparseProbMats<T> = Vec<SparseProbMat<T>>;
pub type SparseProbsHashedIds<T> = HashMap<RnaIdPair, SparseProbMat<T>>;
pub type ProbsHashedIds = HashMap<RnaIdPair, ProbMat>;
pub type ProbsPair = (Probs, Probs);
pub type ProbsPairsHashedIds = HashMap<RnaIdPair, ProbsPair>;
pub type Alignfolds<T, U> = HashMap<RnaIdPair, AlignfoldWrapped<T, U>>;
pub type ProbSets = Vec<Probs>;
pub type SparseScores<T> = HashMap<T, Prob>;
pub type ScoresHashedPoss<T> = HashMap<T, SparseScores<T>>;
pub type AlignShell<T> = HashMap<T, PosPair<T>>;
pub type Seqs = Vec<Seq>;

pub type InputsConsalignWrapped<'a> = (
  &'a mut Pool,
  &'a FastaRecords,
  &'a Path,
  &'a Path,
  Prob,
  Prob,
  ScoreModel,
  TrainType,
  bool,
  Prob,
  Prob,
  bool,
);

pub type InputsTraceback<'a, T, U> = (
  &'a mut AlignfoldWrapped<T, U>,
  &'a AlignfoldPair<'a, T, U>,
  &'a PosQuad<U>,
  &'a ScoreMatsHashedPoss<U>,
  &'a SparseScoreMat<U>,
  &'a mut PosMapsPairs<T>,
  &'a AlignfoldHyperparams,
  &'a AlignShell<U>,
);

pub type InputsRecursiveAlignfold<'a, T> = (
  &'a GuideTree,
  NodeIndex<DefaultIx>,
  &'a SparseProbsHashedIds<T>,
  &'a SparseProbMats<T>,
  &'a FastaRecords,
  &'a AlignfoldHyperparams,
  &'a ProbsPairsHashedIds,
  bool,
);

#[derive(Clone)]
pub struct Align<T> {
  pub pos_map_sets: PosMapSets<T>,
  pub seqs: Seqs,
}

#[derive(Clone)]
pub struct Alignfold<T, U> {
  pub align: Align<T>,
  pub basepairs: SparsePosMat<U>,
  pub unpairs: SparsePoss<U>,
}

impl<T, U> Default for AlignfoldWrapped<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  fn default() -> Self {
    Self::new()
  }
}

impl<T, U> AlignfoldWrapped<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  pub fn new() -> AlignfoldWrapped<T, U> {
    AlignfoldWrapped {
      alignfold: Alignfold::<T, U>::new(),
      rightmost_basepairs_hashed_cols: ColsHashedCols::<U>::default(),
      right_basepairs_hashed_cols: ColSetsHashedCols::<U>::default(),
      rna_ids: RnaIds::new(),
      accuracy: 0.,
    }
  }

  pub fn set_right_basepairs(
    &mut self,
    basepair_prob_mats: &SparseProbMats<T>,
    alignfold_hyperparams: &AlignfoldHyperparams,
  ) {
    let align_len = self.alignfold.align.pos_map_sets.len();
    let num_rnas = self.rna_ids.len();
    for i in 0..align_len {
      let pos_maps = &self.alignfold.align.pos_map_sets[i];
      for j in i + 1..align_len {
        let pos_maps2 = &self.alignfold.align.pos_map_sets[j];
        let pos_map_pairs: Vec<(T, T)> = pos_maps
          .iter()
          .zip(pos_maps2.iter())
          .map(|(&x, &y)| (x, y))
          .collect();
        let mut basepair_prob_sum = 0.;
        for (x, &y) in pos_map_pairs.iter().zip(self.rna_ids.iter()) {
          let y = &basepair_prob_mats[y];
          if let Some(&y) = y.get(x) {
            basepair_prob_sum += y;
          }
        }
        let (i, j) = (
          U::from_usize(i).unwrap() + U::one(),
          U::from_usize(j).unwrap() + U::one(),
        );
        let basepair_prob_avg = basepair_prob_sum / num_rnas as Prob;
        let score = alignfold_hyperparams.param_basepair * basepair_prob_avg - 1.;
        if score >= 0. {
          match self.right_basepairs_hashed_cols.get_mut(&i) {
            Some(x) => {
              x.push((j, score));
            }
            None => {
              let x = vec![(j, score)];
              self.right_basepairs_hashed_cols.insert(i, x);
            }
          }
        }
      }
      let i = U::from_usize(i).unwrap() + U::one();
      if let Some(x) = self.right_basepairs_hashed_cols.get(&i) {
        let x = x.iter().map(|x| x.0).max().unwrap();
        self.rightmost_basepairs_hashed_cols.insert(i, x);
      }
    }
  }

  pub fn set_accuracy(
    &mut self,
    match_probs_hashed_ids: &SparseProbsHashedIds<T>,
    insert_probs_pairs_hashed: &ProbsPairsHashedIds,
  ) {
    let align_len = self.alignfold.align.pos_map_sets.len();
    let num_rnas = self.rna_ids.len();
    let (mut total_expected, mut total): (Score, Score) = (0., 0.);
    for i in 0..num_rnas {
      let rna_id = self.rna_ids[i];
      for j in i + 1..num_rnas {
        let rna_id2 = self.rna_ids[j];
        let rna_id_pair_ordered = if rna_id < rna_id2 {
          (rna_id, rna_id2)
        } else {
          (rna_id2, rna_id)
        };
        let match_probs = &match_probs_hashed_ids[&rna_id_pair_ordered];
        let insert_probs_pair = &insert_probs_pairs_hashed[&rna_id_pair_ordered];
        for k in 0..align_len {
          let pos_maps = &self.alignfold.align.pos_map_sets[k];
          let pos = pos_maps[i];
          let pos2 = pos_maps[j];
          let pos_pair = if rna_id < rna_id2 {
            (pos, pos2)
          } else {
            (pos2, pos)
          };
          if pos_pair.0 != T::zero() || pos_pair.1 != T::zero() {
            if pos_pair.0 != T::zero() && pos_pair.1 != T::zero() {
              if let Some(&x) = match_probs.get(&pos_pair) {
                total_expected += x;
              }
            } else if pos_pair.1 == T::zero() {
              let x = insert_probs_pair.0[pos_pair.0.to_usize().unwrap()];
              total_expected += x;
            } else if pos_pair.0 == T::zero() {
              let x = insert_probs_pair.1[pos_pair.1.to_usize().unwrap()];
              total_expected += x;
            }
            total += 1.;
          }
        }
      }
    }
    let accuracy = total_expected / total;
    self.accuracy = accuracy;
  }

  pub fn sort(&mut self) {
    for x in self.alignfold.align.pos_map_sets.iter_mut() {
      let mut y: Vec<(T, RnaId)> = x
        .iter()
        .zip(self.rna_ids.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
      y.sort_by_key(|y| y.1);
      *x = y.iter().map(|y| y.0).collect();
    }
    self.rna_ids.sort();
  }
}

impl AlignfoldHyperparams {
  pub fn new(x: Score) -> AlignfoldHyperparams {
    AlignfoldHyperparams {
      param_basepair: x,
      param_match: x,
    }
  }
}

impl<T: HashIndex> Default for Align<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: HashIndex> Align<T> {
  pub fn new() -> Align<T> {
    Align {
      pos_map_sets: PosMapSets::<T>::default(),
      seqs: Seqs::new(),
    }
  }
}

impl<T: HashIndex, U: HashIndex> Default for Alignfold<T, U> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: HashIndex, U: HashIndex> Alignfold<T, U> {
  pub fn new() -> Alignfold<T, U> {
    Alignfold {
      align: Align::<T>::new(),
      basepairs: SparsePosMat::<U>::default(),
      unpairs: SparsePoss::<U>::default(),
    }
  }
}

pub const GAP: Char = b'-';
pub const MIN_LOG_HYPERPARAM_BASEPAIR: i32 = 0;
pub const MIN_LOG_HYPERPARAM_MATCH: i32 = MIN_LOG_HYPERPARAM_BASEPAIR;
pub const MAX_LOG_HYPERPARAM_BASEPAIR: i32 = 3;
pub const MAX_LOG_HYPERPARAM_MATCH: i32 = 7;
pub const BRACKET_PAIRS: [(char, char); 9] = [
  ('(', ')'),
  ('<', '>'),
  ('{', '}'),
  ('[', ']'),
  ('A', 'a'),
  ('B', 'b'),
  ('C', 'c'),
  ('D', 'd'),
  ('E', 'e'),
];
pub const DEFAULT_BASEPAIR_PROB_TRAINED: Prob = 2. * DEFAULT_MIN_BASEPAIR_PROB;
pub const DEFAULT_MATCH_PROB_TRAINED: Prob = 2. * DEFAULT_MIN_MATCH_PROB;
pub const DEFAULT_BASEPAIR_PROB_TURNER: Prob = DEFAULT_MIN_BASEPAIR_PROB;
pub const DEFAULT_MATCH_PROB_TURNER: Prob = DEFAULT_MIN_MATCH_PROB;
pub const MIX_COEFF: Prob = 0.5;
pub const HYPERPARAM_ALIFOLD: Prob = 2.;
pub const DEFAULT_SCORE_MODEL: &str = "ensemble";
pub enum ScoreModel {
  Ensemble,
  Turner,
  Trained,
}
pub const OUTPUT_DIR_PATH: &str = "assets/sampled_trnas";

pub fn build_guide_tree<T>(
  fasta_records: &FastaRecords,
  match_probs_hashed_ids: &SparseProbsHashedIds<T>,
  alignfold_hyperparams: &AlignfoldHyperparams,
) -> (GuideTree, NodeIndex<DefaultIx>)
where
  T: HashIndex,
{
  let num_rnas = fasta_records.len();
  let mut guide_tree_scores = GuideTreeScores::default();
  let mut guide_tree = GuideTree::new();
  let mut cluster_sizes = ClusterSizes::default();
  let mut node_indexes = NodeIndexes::default();
  for i in 0..num_rnas {
    for j in i + 1..num_rnas {
      let j = (i, j);
      let match_probs = &match_probs_hashed_ids[&j];
      let x = match_probs
        .values()
        .filter(|&x| alignfold_hyperparams.param_match * x - 1. >= 0.)
        .sum();
      guide_tree_scores.insert(j, x);
    }
  }
  for i in 0..num_rnas {
    let x = guide_tree.add_node(i);
    cluster_sizes.insert(i, 1);
    node_indexes.insert(i, x);
  }
  let mut new_cluster_id = num_rnas;
  while !guide_tree_scores.is_empty() {
    let mut max = NEG_INFINITY;
    let mut argmax = (0, 0);
    for (x, &y) in &guide_tree_scores {
      if y > max {
        argmax = *x;
        max = y;
      }
    }
    let cluster_size_pair = (
      cluster_sizes.remove(&argmax.0).unwrap(),
      cluster_sizes.remove(&argmax.1).unwrap(),
    );
    let new_cluster_size = cluster_size_pair.0 + cluster_size_pair.1;
    cluster_sizes.insert(new_cluster_id, new_cluster_size);
    guide_tree_scores.remove(&argmax);
    for &i in node_indexes.keys() {
      if i == argmax.0 || i == argmax.1 {
        continue;
      }
      let x = if i < argmax.0 {
        (i, argmax.0)
      } else {
        (argmax.0, i)
      };
      let y = guide_tree_scores.remove(&x).unwrap();
      let x = if i < argmax.1 {
        (i, argmax.1)
      } else {
        (argmax.1, i)
      };
      let z = guide_tree_scores.remove(&x).unwrap();
      let z = (cluster_size_pair.0 as Score * y
        + cluster_size_pair.1 as Score * z)
        / new_cluster_size as Score;
      guide_tree_scores.insert((i, new_cluster_id), z);
    }
    let new_node = guide_tree.add_node(new_cluster_id);
    node_indexes.insert(new_cluster_id, new_node);
    let edge_len = max / 2.;
    let argmax_node_pair = (
      node_indexes.remove(&argmax.0).unwrap(),
      node_indexes.remove(&argmax.1).unwrap(),
    );
    guide_tree.add_edge(new_node, argmax_node_pair.0, edge_len);
    guide_tree.add_edge(new_node, argmax_node_pair.1, edge_len);
    new_cluster_id += 1;
  }
  let root = node_indexes[&(new_cluster_id - 1)];
  (guide_tree, root)
}

pub fn consalign<T, U>(
  fasta_records: &FastaRecords,
  match_probs_hashed_ids: &SparseProbsHashedIds<T>,
  basepair_prob_mats: &SparseProbMats<T>,
  alignfold_hyperparams: &AlignfoldHyperparams,
  insert_probs_pairs_hashed: &ProbsPairsHashedIds,
) -> AlignfoldWrapped<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let (guide_tree, root) = build_guide_tree(
    fasta_records,
    match_probs_hashed_ids,
    alignfold_hyperparams,
  );
  let ends_alignfold = true;
  recursive_alignfold((
    &guide_tree,
    root,
    match_probs_hashed_ids,
    basepair_prob_mats,
    fasta_records,
    alignfold_hyperparams,
    insert_probs_pairs_hashed,
    ends_alignfold,
  ))
}

pub fn recursive_alignfold<T, U>(
  inputs: InputsRecursiveAlignfold<T>,
) -> AlignfoldWrapped<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let (
    guide_tree,
    node,
    match_probs_hashed_ids,
    basepair_prob_mats,
    fasta_records,
    alignfold_hyperparams,
    insert_probs_pairs_hashed,
    ends_alignfold,
  ) = inputs;
  let num_rnas = fasta_records.len();
  let rna_id = *guide_tree.node_weight(node).unwrap();
  if rna_id < num_rnas {
    let x = &fasta_records[rna_id].seq;
    seq2alignfold(x, rna_id, basepair_prob_mats, alignfold_hyperparams)
  } else {
    let ends_alignfold2 = false;
    let mut neighbors = guide_tree.neighbors_directed(node, Outgoing).detach();
    let x = neighbors.next_node(guide_tree).unwrap();
    let y = neighbors.next_node(guide_tree).unwrap();
    let x = recursive_alignfold((
      guide_tree,
      x,
      match_probs_hashed_ids,
      basepair_prob_mats,
      fasta_records,
      alignfold_hyperparams,
      insert_probs_pairs_hashed,
      ends_alignfold2,
    ));
    let y = recursive_alignfold((
      guide_tree,
      y,
      match_probs_hashed_ids,
      basepair_prob_mats,
      fasta_records,
      alignfold_hyperparams,
      insert_probs_pairs_hashed,
      ends_alignfold2,
    ));
    merge_alignfolds(
      &(&x, &y),
      match_probs_hashed_ids,
      basepair_prob_mats,
      alignfold_hyperparams,
      insert_probs_pairs_hashed,
      ends_alignfold,
    )
  }
}

pub fn seq2alignfold<T, U>(
  x: &Seq,
  y: RnaId,
  z: &SparseProbMats<T>,
  a: &AlignfoldHyperparams,
) -> AlignfoldWrapped<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let mut b = AlignfoldWrapped::new();
  b.alignfold.align.pos_map_sets = (1..x.len() - 1)
    .map(|x| vec![T::from_usize(x).unwrap()])
    .collect();
  b.rna_ids = vec![y];
  b.set_right_basepairs(z, a);
  b
}

pub fn merge_alignfolds<T, U>(
  alignfold_pair: &AlignfoldPair<T, U>,
  match_probs_hashed_ids: &SparseProbsHashedIds<T>,
  basepair_prob_mats: &SparseProbMats<T>,
  alignfold_hyperparams: &AlignfoldHyperparams,
  insert_probs_pairs_hashed: &ProbsPairsHashedIds,
  ends_alignfold: bool,
) -> AlignfoldWrapped<T, U>
where
  T: HashIndex,
  U: HashIndex,
{
  let align_len_pair = (
    alignfold_pair
      .0
      .alignfold
      .align
      .pos_map_sets
      .len(),
    alignfold_pair
      .1
      .alignfold
      .align
      .pos_map_sets
      .len(),
  );
  let rna_num_pair = (
    alignfold_pair.0.rna_ids.len(),
    alignfold_pair.1.rna_ids.len(),
  );
  let num_rnas = rna_num_pair.0 + rna_num_pair.1;
  let denom = (rna_num_pair.0 * rna_num_pair.1) as Prob;
  let mut match_scores = SparseScoreMat::<U>::default();
  let rna_ids = &alignfold_pair.0.rna_ids;
  let rna_ids2 = &alignfold_pair.1.rna_ids;
  let pos_map_sets = &alignfold_pair.0.alignfold.align.pos_map_sets;
  let pos_map_sets2 = &alignfold_pair.1.alignfold.align.pos_map_sets;
  let align_len_pair = (
    U::from_usize(align_len_pair.0).unwrap(),
    U::from_usize(align_len_pair.1).unwrap(),
  );
  let pseudo_col_quad = (
    U::zero(),
    align_len_pair.0 + U::one(),
    U::zero(),
    align_len_pair.1 + U::one(),
  );
  let mut align_shell = AlignShell::<U>::default();
  for i in range_inclusive(U::one(), align_len_pair.0) {
    let long_i = i.to_usize().unwrap();
    let pos_maps = &pos_map_sets[long_i - 1];
    for j in range_inclusive(U::one(), align_len_pair.1) {
      let col_pair = (i, j);
      let long_j = j.to_usize().unwrap();
      let mut match_prob_sum = 0.;
      let pos_maps2 = &pos_map_sets2[long_j - 1];
      for (&x, &y) in rna_ids.iter().zip(pos_maps.iter()) {
        for (&z, &a) in rna_ids2.iter().zip(pos_maps2.iter()) {
          let b = if x < z {
            (x, z)
          } else {
            (z, x)
          };
          let c = &match_probs_hashed_ids[&b];
          let d = if x < z {
            (y, a)
          } else {
            (a, y)
          };
          if let Some(&c) = c.get(&d) {
            match_prob_sum += c;
          }
        }
      }
      let match_score = alignfold_hyperparams.param_match * match_prob_sum / denom - 1.;
      if match_score >= 0. {
        match_scores.insert(col_pair, match_score);
        match align_shell.get_mut(&(i - U::one())) {
          Some(x) => {
            x.0 = x.0.min(j - U::one());
            x.1 = x.1.max(j - U::one());
          }
          None => {
            align_shell.insert(i - U::one(), (j - U::one(), j - U::one()));
          }
        }
        match align_shell.get_mut(&i) {
          Some(x) => {
            x.0 = x.0.min(j);
            x.1 = x.1.max(j);
          }
          None => {
            align_shell.insert(i, (j, j));
          }
        }
      }
    }
  }
  for i in range_inclusive(U::zero(), align_len_pair.0).rev() {
    let x = if align_shell.contains_key(&(i + U::one())) {
      align_shell[&(i + U::one())].0
    } else {
      U::zero()
    };
    match align_shell.get_mut(&i) {
      Some(y) => {
        y.0 = y.0.min(x);
      }
      None => {
        align_shell.insert(i, (x, U::zero()));
      }
    }
  }
  for i in range_inclusive(U::one(), align_len_pair.0) {
    let x = if align_shell.contains_key(&(i - U::one())) {
      align_shell[&(i - U::one())].1
    } else {
      U::zero()
    };
    match align_shell.get_mut(&i) {
      Some(y) => {
        y.1 = y.1.max(x);
      }
      None => {
        align_shell.insert(i, (U::zero(), x));
      }
    }
  }
  for i in range_inclusive(U::one(), align_len_pair.0) {
    if let Some(x) = align_shell.get_mut(&i) {
      x.0 = x.0.min(x.1);
    }
  }
  align_shell.get_mut(&U::zero()).unwrap().0 = U::zero();
  align_shell.get_mut(&align_len_pair.0).unwrap().1 = align_len_pair.1;
  let mut scores_hashed_cols = ScoreMatsHashedPoss::default();
  for i in range_inclusive(U::one(), align_len_pair.0).rev() {
    if let Some(&j) = alignfold_pair.0.rightmost_basepairs_hashed_cols.get(&i) {
      for k in range_inclusive(U::one(), align_len_pair.1).rev() {
        let x = (i, k);
        if !match_scores.contains_key(&x) {
          continue;
        }
        if let Some(&l) = alignfold_pair.1.rightmost_basepairs_hashed_cols.get(&k) {
          let y = (i, j, k, l);
          let scores = get_scores(
            &scores_hashed_cols,
            &match_scores,
            &y,
            &align_shell,
          );
          update_scores_hashed_cols(
            &mut scores_hashed_cols,
            &x,
            alignfold_pair,
            &scores,
            &match_scores,
          );
        }
      }
    }
  }
  let mut new_alignfold = AlignfoldWrapped::new();
  let mut new_rna_ids = alignfold_pair.0.rna_ids.clone();
  let mut rna_ids_append = alignfold_pair.1.rna_ids.clone();
  new_rna_ids.append(&mut rna_ids_append);
  new_alignfold.rna_ids = new_rna_ids;
  let mut basepairs = PosMapsPairs::<T>::new();
  traceback((
    &mut new_alignfold,
    alignfold_pair,
    &pseudo_col_quad,
    &scores_hashed_cols,
    &match_scores,
    &mut basepairs,
    alignfold_hyperparams,
    &align_shell,
  ));
  let align_len = new_alignfold
    .alignfold
    .align
    .pos_map_sets
    .len();
  let pos_maps_gapped_only = vec![T::zero(); num_rnas];
  for i in (0..align_len).rev() {
    let x = &new_alignfold.alignfold.align.pos_map_sets[i];
    if *x == pos_maps_gapped_only {
      new_alignfold
        .alignfold
        .align
        .pos_map_sets
        .remove(i);
    }
  }
  if ends_alignfold {
    let align_len = new_alignfold
      .alignfold
      .align
      .pos_map_sets
      .len();
    for x in &basepairs {
      for i in 0..align_len {
        let y = &new_alignfold.alignfold.align.pos_map_sets[i];
        if *y != x.0 {
          continue;
        }
        let short_i = U::from_usize(i).unwrap();
        for j in i + 1..align_len {
          let y = &new_alignfold.alignfold.align.pos_map_sets[j];
          if *y == x.1 {
            let short_j = U::from_usize(j).unwrap();
            new_alignfold
              .alignfold
              .basepairs
              .insert((short_i, short_j));
            break;
          }
        }
      }
    }
    new_alignfold.set_accuracy(
      match_probs_hashed_ids,
      insert_probs_pairs_hashed,
    );
  } else {
    new_alignfold.set_right_basepairs(basepair_prob_mats, alignfold_hyperparams);
  }
  new_alignfold
}

pub fn get_scores<T>(
  scores_hashed_cols: &ScoreMatsHashedPoss<T>,
  match_scores: &SparseProbMat<T>,
  col_quad: &PosQuad<T>,
  align_shell: &AlignShell<T>,
) -> SparseProbMat<T>
where
  T: HashIndex,
{
  let (i, j, k, l) = *col_quad;
  let mut scores = SparseProbMat::<T>::default();
  for u in range(i, j) {
    let (begin, end) = align_shell[&u];
    let begin = begin.max(k);
    let end = end.min(l - T::one());
    for v in range_inclusive(begin, end) {
      let col_pair = (u, v);
      if u == i && v == k {
        scores.insert(col_pair, 0.);
        continue;
      }
      let mut score = NEG_INFINITY;
      if let Some(x) = scores_hashed_cols.get(&col_pair) {
        for (x, y) in x {
          let z = (x.0 - T::one(), x.1 - T::one());
          if !(i < x.0 && k < x.1) {
            continue;
          }
          if let Some(&z) = scores.get(&z) {
            let z = y + z;
            if z > score {
              score = z;
            }
          }
        }
      }
      let col_pair_match = (u - T::one(), v - T::one());
      if let Some(&x) = match_scores.get(&col_pair) {
        if let Some(&y) = scores.get(&col_pair_match) {
          let y = x + y;
          if y > score {
            score = y;
          }
        }
      }
      let col_pair_insert = (u - T::one(), v);
      if let Some(&x) = scores.get(&col_pair_insert) {
        if x > score {
          score = x;
        }
      }
      let col_pair_del = (u, v - T::one());
      if let Some(&x) = scores.get(&col_pair_del) {
        if x > score {
          score = x;
        }
      }
      if score > NEG_INFINITY {
        scores.insert(col_pair, score);
      }
    }
  }
  scores
}

pub fn update_scores_hashed_cols<T, U>(
  scores_hashed_cols: &mut ScoreMatsHashedPoss<U>,
  col_pair_left: &PosPair<U>,
  alignfold_pair: &AlignfoldPair<T, U>,
  scores: &SparseProbMat<U>,
  match_scores: &SparseProbMat<U>,
) where
  T: HashIndex,
  U: HashIndex,
{
  let (i, k) = *col_pair_left;
  let right_basepairs = &alignfold_pair.0.right_basepairs_hashed_cols[&i];
  let right_basepairs2 = &alignfold_pair.1.right_basepairs_hashed_cols[&k];
  let match_score_left = match_scores[col_pair_left];
  for &(j, x) in right_basepairs.iter() {
    for &(l, y) in right_basepairs2.iter() {
      let z = (j, l);
      if !match_scores.contains_key(&z) {
        continue;
      }
      let a = match_scores[&z];
      let a = match_score_left + x + y + a;
      let a = a + scores[&(j - U::one(), l - U::one())];
      match scores_hashed_cols.get_mut(&z) {
        Some(x) => {
          x.insert(*col_pair_left, a);
        }
        None => {
          let mut x = SparseScoreMat::default();
          x.insert(*col_pair_left, a);
          scores_hashed_cols.insert(z, x);
        }
      }
    }
  }
}

pub fn traceback<T, U>(inputs: InputsTraceback<T, U>)
where
  T: HashIndex,
  U: HashIndex,
{
  let (
    new_alignfold,
    alignfold_pair,
    col_quad,
    scores_hashed_cols,
    match_scores,
    basepair_pos_maps_pairs,
    alignfold_hyperparams,
    align_shell,
  ) = inputs;
  let rna_num_pair = (
    alignfold_pair.0.rna_ids.len(),
    alignfold_pair.1.rna_ids.len(),
  );
  let mut score;
  let scores = get_scores(
    scores_hashed_cols,
    match_scores,
    col_quad,
    align_shell,
  );
  let (i, j, k, l) = *col_quad;
  let (mut u, mut v) = (j - U::one(), l - U::one());
  while u > i || v > k {
    let col_pair = (u, v);
    score = scores[&col_pair];
    let (long_u, long_v) = (u.to_usize().unwrap(), v.to_usize().unwrap());
    if u > i && v > k {
      if let Some(&x) = match_scores.get(&col_pair) {
        let y = (u - U::one(), v - U::one());
        let y = scores[&y] + x;
        if y == score {
          let mut z =
            alignfold_pair.0.alignfold.align.pos_map_sets[long_u - 1].clone();
          let mut a =
            alignfold_pair.1.alignfold.align.pos_map_sets[long_v - 1].clone();
          z.append(&mut a);
          new_alignfold
            .alignfold
            .align
            .pos_map_sets
            .insert(0, z);
          u = u - U::one();
          v = v - U::one();
          continue;
        }
      }
      let mut found_pairmatch = false;
      if let Some(x) = scores_hashed_cols.get(&col_pair) {
        for (x, y) in x {
          if !(i < x.0 && k < x.1) {
            continue;
          }
          let z = (x.0 - U::one(), x.1 - U::one());
          if let Some(&a) = scores.get(&z) {
            let y = a + y;
            if y == score {
              let mut y =
                alignfold_pair.0.alignfold.align.pos_map_sets[long_u - 1].clone();
              let mut a =
                alignfold_pair.1.alignfold.align.pos_map_sets[long_v - 1].clone();
              y.append(&mut a);
              new_alignfold
                .alignfold
                .align
                .pos_map_sets
                .insert(0, y.clone());
              traceback((
                new_alignfold,
                alignfold_pair,
                &(x.0, u, x.1, v),
                scores_hashed_cols,
                match_scores,
                basepair_pos_maps_pairs,
                alignfold_hyperparams,
                align_shell,
              ));
              let x = (
                x.0.to_usize().unwrap(),
                x.1.to_usize().unwrap(),
              );
              let mut a = alignfold_pair.0.alignfold.align.pos_map_sets
                [x.0 - 1]
                .clone();
              let mut b = alignfold_pair.1.alignfold.align.pos_map_sets
                [x.1 - 1]
                .clone();
              a.append(&mut b);
              new_alignfold
                .alignfold
                .align
                .pos_map_sets
                .insert(0, a.clone());
              basepair_pos_maps_pairs.push((a, y));
              u = z.0;
              v = z.1;
              found_pairmatch = true;
              break;
            }
          }
        }
      }
      if found_pairmatch {
        continue;
      }
    }
    if u > i {
      if let Some(&x) = scores.get(&(u - U::one(), v)) {
        if x == score {
          let mut x =
            alignfold_pair.0.alignfold.align.pos_map_sets[long_u - 1].clone();
          let mut y = vec![T::zero(); rna_num_pair.1];
          x.append(&mut y);
          new_alignfold
            .alignfold
            .align
            .pos_map_sets
            .insert(0, x);
          u = u - U::one();
          continue;
        }
      }
    }
    if v > k {
      if let Some(&x) = scores.get(&(u, v - U::one())) {
        if x == score {
          let mut x = vec![T::zero(); rna_num_pair.0];
          let mut y =
            alignfold_pair.1.alignfold.align.pos_map_sets[long_v - 1].clone();
          x.append(&mut y);
          new_alignfold
            .alignfold
            .align
            .pos_map_sets
            .insert(0, x);
          v = v - U::one();
        }
      }
    }
  }
}

pub fn base2char(c: Base) -> Char {
  match c {
    A => A_UPPER,
    C => C_UPPER,
    G => G_UPPER,
    U => U_UPPER,
    _ => {
      panic!();
    }
  }
}

pub fn consalifold<T>(
  basepair_probs_mix: &SparseProbMat<T>,
  align_len: T,
  hyperparam_alifold: Score,
) -> SparsePosMat<T>
where
  T: HashIndex,
{
  let basepair_weights: SparseProbMat<T> = basepair_probs_mix
    .iter()
    .filter(|x| hyperparam_alifold * x.1 - 1. >= 0.)
    .map(|x| (*x.0, *x.1))
    .collect();
  let mut scores_hashed_cols = ScoresHashedPoss::default();
  let mut right_basepairs_hashed_cols = ColSetsHashedCols::<T>::default();
  for (x, &y) in &basepair_weights {
    match right_basepairs_hashed_cols.get_mut(&x.0) {
      Some(z) => {
        z.push((x.1, y));
      }
      None => {
        let z = vec![(x.1, y)];
        right_basepairs_hashed_cols.insert(x.0, z);
      }
    }
  }
  let mut rightmost_basepairs_hashed_cols = ColsHashedCols::<T>::default();
  for (&x, y) in &right_basepairs_hashed_cols {
    let y = y.iter().map(|y| y.0).max().unwrap();
    rightmost_basepairs_hashed_cols.insert(x, y);
  }
  for i in range_inclusive(T::one(), align_len).rev() {
    if let Some(&j) = rightmost_basepairs_hashed_cols.get(&i) {
      let j = (i, j);
      let scores = get_scores_alifold(&scores_hashed_cols, &j);
      update_scores_hashed_alifold(
        &mut scores_hashed_cols,
        j.0,
        &scores,
        &right_basepairs_hashed_cols,
      );
    }
  }
  let pseudo_col_pair = (T::zero(), align_len + T::one());
  let mut basepairs = SparsePosMat::<T>::default();
  traceback_alifold(&mut basepairs, &pseudo_col_pair, &scores_hashed_cols);
  basepairs
}

pub fn traceback_alifold<T>(
  x: &mut SparsePosMat<T>,
  y: &PosPair<T>,
  z: &ScoresHashedPoss<T>,
) where
  T: HashIndex,
{
  let mut a;
  let b = get_scores_alifold(z, y);
  let (i, j) = *y;
  let mut k = j - T::one();
  while k > i {
    a = b[&k];
    let c = b[&(k - T::one())];
    if c == a {
      k = k - T::one();
    }
    if let Some(c) = z.get(&k) {
      for (&c, d) in c {
        if i >= c {
          continue;
        }
        let e = c - T::one();
        let b = b[&e];
        let d = b + d;
        if d == a {
          let d = (c, k);
          traceback_alifold(x, &d, z);
          let d = (c - T::one(), k - T::one());
          x.insert(d);
          k = e;
          break;
        }
      }
    }
  }
}

pub fn update_scores_hashed_alifold<T>(
  x: &mut ScoresHashedPoss<T>,
  y: T,
  z: &SparseScores<T>,
  a: &ColSetsHashedCols<T>,
) where
  T: HashIndex,
{
  let a = &a[&y];
  for &(a, b) in a {
    let b = b + z[&(a - T::one())];
    match x.get_mut(&a) {
      Some(x) => {
        x.insert(y, b);
      }
      None => {
        let mut z = SparseScores::default();
        z.insert(y, b);
        x.insert(a, z);
      }
    }
  }
}

pub fn get_scores_alifold<T>(x: &ScoresHashedPoss<T>, y: &PosPair<T>) -> SparseScores<T>
where
  T: HashIndex,
{
  let (i, j) = *y;
  let mut z = SparseScores::<T>::default();
  for k in range(i, j) {
    if k == i {
      z.insert(k, 0.);
      continue;
    }
    let mut y = z[&(k - T::one())];
    if let Some(x) = x.get(&k) {
      for (&l, x) in x {
        if i >= l {
          continue;
        }
        let z = z[&(l - T::one())];
        let z = x + z;
        if z > y {
          y = z;
        }
      }
    }
    z.insert(k, y);
  }
  z
}

pub fn run_command(x: &str, y: &[&str], z: &str) -> Output {
  Command::new(x).args(y).output().expect(z)
}

pub fn get_basepair_probs_alifold<T, U>(
  alignfold: &AlignfoldWrapped<T, U>,
  align_file_path: &Path,
  fasta_records: &FastaRecords,
  output_dir_path: &Path,
) -> SparseProbMat<U>
where
  T: HashIndex,
  U: HashIndex,
{
  let cwd = env::current_dir().unwrap();
  let align_file_path = cwd.join(align_file_path);
  let mut writer = BufWriter::new(File::create(align_file_path.clone()).unwrap());
  let mut buf = "CLUSTAL format sequence alignment\n\n".to_string();
  let align_len = alignfold.alignfold.align.pos_map_sets.len();
  let fasta_ids: Vec<FastaId> = alignfold
    .rna_ids
    .iter()
    .map(|&x| fasta_records[x].fasta_id.clone())
    .collect();
  let max_seq_id_len = fasta_ids.iter().map(|x| x.len()).max().unwrap();
  for (i, &rna_id) in alignfold.rna_ids.iter().enumerate() {
    let seq_id = &fasta_records[rna_id].fasta_id;
    buf.push_str(seq_id);
    let mut clustal_row = vec![b' '; max_seq_id_len - seq_id.len() + 2];
    let seq = &alignfold.alignfold.align.seqs[i];
    let mut align_row = (0..align_len)
      .map(|x| {
        let x = alignfold.alignfold.align.pos_map_sets[x][i]
          .to_usize()
          .unwrap();
        if x == 0 {
          GAP
        } else {
          base2char(seq[x])
        }
      })
      .collect::<Vec<Char>>();
    clustal_row.append(&mut align_row);
    let clustal_row = unsafe { from_utf8_unchecked(&clustal_row) };
    buf.push_str(clustal_row);
    buf.push('\n');
  }
  let _ = writer.write_all(buf.as_bytes());
  let _ = writer.flush();
  let align_file_prefix = align_file_path.file_stem().unwrap().to_str().unwrap();
  let arg = format!("--id-prefix={align_file_prefix}");
  let args = vec![
    "-p",
    align_file_path.to_str().unwrap(),
    &arg,
    "--noPS",
    "--noDP",
  ];
  let _ = env::set_current_dir(output_dir_path);
  let _ = run_command("RNAalifold", &args, "Failed to run RNAalifold");
  let _ = env::set_current_dir(cwd);
  let mut basepair_probs_alifold = SparseProbMat::<U>::default();
  let output_file_path = output_dir_path.join(String::from(align_file_prefix) + "_0001_ali.out");
  let output_file = BufReader::new(File::open(output_file_path.clone()).unwrap());
  for (x, y) in output_file.lines().enumerate() {
    if x == 0 {
      continue;
    }
    let y = y.unwrap();
    if !y.starts_with(' ') {
      continue;
    }
    let y: Vec<&str> = y.split_whitespace().collect();
    let z = (
      U::from_usize(y[0].parse().unwrap()).unwrap(),
      U::from_usize(y[1].parse().unwrap()).unwrap(),
    );
    let mut a = String::from(y[3]);
    a.pop();
    let a = 0.01 * a.parse::<Prob>().unwrap();
    if a == 0. {
      continue;
    }
    basepair_probs_alifold.insert(z, a);
  }
  let _ = remove_file(align_file_path);
  let _ = remove_file(output_file_path);
  basepair_probs_alifold
}

pub fn get_basepair_probs_mix<T, U>(
  alignfold: &AlignfoldWrapped<T, U>,
  basepair_prob_mats: &SparseProbMats<T>,
  basepair_probs_alifold: &SparseProbMat<U>,
  disables_alifold: bool,
) -> SparseProbMat<U>
where
  T: HashIndex,
  U: HashIndex,
{
  let mut basepair_probs_mix = SparseProbMat::<U>::default();
  let align_len = alignfold.alignfold.align.pos_map_sets.len();
  let num_rnas = alignfold.rna_ids.len();
  for i in 0..align_len {
    let pos_maps = &alignfold.alignfold.align.pos_map_sets[i];
    let short_i = U::from_usize(i).unwrap();
    for j in i + 1..align_len {
      let short_j = U::from_usize(j).unwrap();
      let pos_pair = (short_i + U::one(), short_j + U::one());
      let basepair_prob_alifold = if disables_alifold {
        NEG_INFINITY
      } else {
        match basepair_probs_alifold.get(&pos_pair) {
          Some(&x) => x,
          None => 0.,
        }
      };
      let pos_maps2 = &alignfold.alignfold.align.pos_map_sets[j];
      let pos_map_pairs: Vec<(T, T)> = pos_maps
        .iter()
        .zip(pos_maps2.iter())
        .map(|(&x, &y)| (x, y))
        .collect();
      let mut basepair_prob_sum = 0.;
      for (x, &y) in pos_map_pairs.iter().zip(alignfold.rna_ids.iter()) {
        let y = &basepair_prob_mats[y];
        if let Some(&x) = y.get(x) {
          basepair_prob_sum += x;
        }
      }
      let basepair_prob_avg = basepair_prob_sum / num_rnas as Prob;
      let basepair_prob_mix = if disables_alifold {
        basepair_prob_avg
      } else {
        MIX_COEFF * basepair_prob_avg + (1. - MIX_COEFF) * basepair_prob_alifold
      };
      if basepair_prob_avg > 0. {
        basepair_probs_mix.insert(pos_pair, basepair_prob_mix);
      }
    }
  }
  basepair_probs_mix
}

pub fn get_insert_probs_pair<T>(
  x: &SparseProbMat<T>,
  y: &(usize, usize),
) -> ProbsPair
where
  T: HashIndex,
{
  let mut z = (vec![1.; y.0], vec![1.; y.1]);
  for (&(i, j), a) in x {
    let (i, j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
    z.0[i] -= a;
    z.1[j] -= a;
  }
  z
}

pub fn consalign_wrapped<T, U>(
  inputs: InputsConsalignWrapped,
) -> (AlignfoldWrapped<T, U>, AlignfoldHyperparams)
where
  T: HashIndex,
  U: HashIndex,
{
  let (
    thread_pool,
    fasta_records,
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
  ) = inputs;
  let mut align_scores = AlignScores::new(0.);
  if disables_transplant {
    align_scores.transfer();
  } else {
    copy_alignfold_scores_align(
      &mut align_scores,
      &AlignfoldScores::load_trained_scores(),
    );
  }
  let seqs = fasta_records.iter().map(|x| &x.seq[..]).collect();
  let produces_context_profs = false;
  let produces_match_probs = true;
  let (alignfold_prob_mats_turner, match_probs_hashed_turner) =
    if matches!(score_model, ScoreModel::Ensemble)
      || matches!(score_model, ScoreModel::Turner)
    {
      consprob::<T>(
        thread_pool,
        &seqs,
        min_basepair_prob_turner,
        min_match_prob_turner,
        produces_context_profs,
        produces_match_probs,
        &align_scores,
      )
    } else {
      (
        ProbMatSetsAvg::<T>::default(),
        MatchProbsHashedIds::<T>::default(),
      )
    };
  let match_probs_hashed_turner: SparseProbsHashedIds<T> =
    match_probs_hashed_turner
      .iter()
      .map(|(x, y)| (*x, y.match_probs.clone()))
      .collect();
  let basepair_prob_mats_turner: SparseProbMats<T> = alignfold_prob_mats_turner
    .iter()
    .map(|x| x.basepair_probs.clone())
    .collect();
  drop(alignfold_prob_mats_turner);
  // drop(align_prob_mat_pairs_with_rna_id_pairs_turner);
  let (alignfold_prob_mats_trained, match_probs_hashed_trained) =
    if matches!(score_model, ScoreModel::Ensemble)
      || matches!(score_model, ScoreModel::Trained)
    {
      consprob_trained::<T>(
        thread_pool,
        &seqs,
        min_basepair_prob_trained,
        min_match_prob_trained,
        produces_context_profs,
        produces_match_probs,
        train_type,
      )
    } else {
      (
        ProbMatSetsAvg::<T>::default(),
        MatchProbsHashedIds::<T>::default(),
      )
    };
  let match_probs_hashed_trained: SparseProbsHashedIds<T> =
    match_probs_hashed_trained
      .iter()
      .map(|(x, y)| (*x, y.match_probs.clone()))
      .collect();
  let basepair_prob_mats_trained: SparseProbMats<T> = alignfold_prob_mats_trained
    .iter()
    .map(|x| x.basepair_probs.clone())
    .collect();
  drop(alignfold_prob_mats_trained);
  // drop(align_prob_mat_pairs_with_rna_id_pairs_trained);
  let num_fasta_records = fasta_records.len();
  let mut basepair_prob_mats_fused = vec![SparseProbMat::<T>::new(); num_fasta_records];
  let mut match_probs_hashed_fused = SparseProbsHashedIds::<T>::default();
  let mut insert_probs_pairs_hashed = ProbsPairsHashedIds::default();
  for x in 0..num_fasta_records {
    for y in x + 1..num_fasta_records {
      let y = (x, y);
      match_probs_hashed_fused.insert(y, SparseProbMat::<T>::default());
      insert_probs_pairs_hashed.insert(y, (Probs::new(), Probs::new()));
    }
  }
  if matches!(score_model, ScoreModel::Ensemble) {
    thread_pool.scoped(|x| {
      for (y, z, a) in multizip((
        basepair_prob_mats_fused.iter_mut(),
        basepair_prob_mats_turner.iter(),
        basepair_prob_mats_trained.iter(),
      )) {
        x.execute(move || {
          *y = z
            .iter()
            .map(|(b, &z)| (*b, 0.5 * z))
            .collect();
          for (b, a) in a {
            let a = 0.5 * a;
            match y.get_mut(b) {
              Some(y) => {
                *y += a;
              }
              None => {
                y.insert(*b, a);
              }
            }
          }
        });
      }
    });
    thread_pool.scoped(|x| {
      for (y, z) in match_probs_hashed_fused.iter_mut()
      {
        let match_probs_turner = &match_probs_hashed_turner[y];
        let match_probs_trained = &match_probs_hashed_trained[y];
        x.execute(move || {
          *z = match_probs_turner
            .iter()
            .map(|(a, &b)| (*a, 0.5 * b))
            .collect();
          for (a, b) in match_probs_trained {
            let b = 0.5 * b;
            match z.get_mut(a) {
              Some(z) => {
                *z += b;
              }
              None => {
                z.insert(*a, b);
              }
            }
          }
        });
      }
    });
  } else if matches!(score_model, ScoreModel::Turner) {
    basepair_prob_mats_fused = basepair_prob_mats_turner.clone();
    match_probs_hashed_fused = match_probs_hashed_turner.clone();
  } else {
    basepair_prob_mats_fused = basepair_prob_mats_trained.clone();
    match_probs_hashed_fused = match_probs_hashed_trained.clone();
  }
  thread_pool.scoped(|x| {
    for (y, z) in insert_probs_pairs_hashed.iter_mut() {
      let a = &match_probs_hashed_fused[y];
      let y = (
        fasta_records[y.0].seq.len(),
        fasta_records[y.1].seq.len(),
      );
      x.execute(move || {
        *z = get_insert_probs_pair(a, &y);
      });
    }
  });
  if !output_dir_path.exists() {
    let _ = create_dir(output_dir_path);
  }
  let input_file_prefix = input_file_path.file_stem().unwrap().to_str().unwrap();
  let align_file_path = output_dir_path.join(&format!("{input_file_prefix}.aln"));
  let mut alignfold_candidates = Vec::new();
  for x in MIN_LOG_HYPERPARAM_MATCH..MAX_LOG_HYPERPARAM_MATCH + 1 {
    let x = (2. as Prob).powi(x) + 1.;
    for y in MIN_LOG_HYPERPARAM_BASEPAIR..MAX_LOG_HYPERPARAM_BASEPAIR + 1 {
      let y = (2. as Prob).powi(y) + 1.;
      let mut z = AlignfoldHyperparams::new(x);
      z.param_basepair = y;
      alignfold_candidates.push((z, AlignfoldWrapped::new()));
    }
  }
  thread_pool.scoped(|x| {
    let y = &match_probs_hashed_fused;
    let z = &basepair_prob_mats_fused;
    let a = &insert_probs_pairs_hashed;
    for b in &mut alignfold_candidates {
      x.execute(move || {
        b.1 = consalign::<T, U>(
          fasta_records,
          y,
          z,
          &b.0,
          a,
        );
      });
    }
  });
  let mut alignfold_final = AlignfoldWrapped::new();
  let mut alignfold_hyperparams = AlignfoldHyperparams::new(0.);
  for x in &alignfold_candidates {
    let y = &x.1;
    if y.accuracy > alignfold_final.accuracy {
      alignfold_hyperparams = x.0.clone();
      alignfold_final = y.clone();
    }
  }
  alignfold_final.sort();
  alignfold_final.alignfold.align.seqs = fasta_records.iter().map(|x| x.seq.clone()).collect();
  let basepair_probs_alifold = if disables_alifold {
    SparseProbMat::<U>::default()
  } else {
    get_basepair_probs_alifold(&alignfold_final, &align_file_path, fasta_records, output_dir_path)
  };
  let basepair_probs_mix = get_basepair_probs_mix(&alignfold_final, &basepair_prob_mats_fused, &basepair_probs_alifold, disables_alifold);
  let align_len = alignfold_final.alignfold.align.pos_map_sets.len();
  let align_len = U::from_usize(align_len).unwrap();
  alignfold_final.alignfold.basepairs =
    consalifold(&basepair_probs_mix, align_len, HYPERPARAM_ALIFOLD);
  (alignfold_final, alignfold_hyperparams)
}
