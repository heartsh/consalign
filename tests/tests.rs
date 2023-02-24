extern crate consalign;

use consalign::*;

#[test]
fn test_consalign() {
  let input_file_path = Path::new(EXAMPLE_FASTA_FILE_PATH);
  let fasta_file_reader = Reader::from_file(input_file_path).unwrap();
  let mut fasta_records = FastaRecords::new();
  let mut max_seq_len = 0;
  for fasta_record in fasta_file_reader.records() {
    let fasta_record = fasta_record.unwrap();
    let mut seq = bytes2seq(fasta_record.seq());
    seq.insert(0, PSEUDO_BASE);
    seq.push(PSEUDO_BASE);
    let seq_len = seq.len();
    if seq_len > max_seq_len {
      max_seq_len = seq_len;
    }
    fasta_records.push(FastaRecord::new(String::from(fasta_record.id()), seq));
  }
  let num_threads = num_cpus::get() as NumThreads;
  let mut thread_pool = Pool::new(num_threads);
  let output_dir_path = Path::new(OUTPUT_DIR_PATH);
  let disables_alifold = true;
  let disables_transplant = false;
  let (alignfold, alignfold_hyperparams) = consalign_wrapped::<u8, u8>((
    &mut thread_pool,
    &fasta_records,
    output_dir_path,
    input_file_path,
    DEFAULT_BASEPAIR_PROB_TRAINED,
    DEFAULT_MATCH_PROB_TRAINED,
    ScoreModel::Ensemble,
    TrainType::TrainedTransfer,
    disables_alifold,
    DEFAULT_BASEPAIR_PROB_TURNER,
    DEFAULT_MATCH_PROB_TURNER,
    disables_transplant,
  ));
  let num_rnas = alignfold.rna_ids.len();
  let pos_maps_gapped_only = vec![0; num_rnas];
  for (i, x) in alignfold.alignfold.align.pos_map_sets.iter().enumerate() {
    assert!(*x != pos_maps_gapped_only);
    if i == 0 {
      continue;
    }
    let y = &alignfold.alignfold.align.pos_map_sets[i - 1];
    for (&x, &y) in x.iter().zip(y.iter()) {
      assert!(x > y || x == 0);
    }
  }
  assert!(alignfold_hyperparams.param_match > 1. && alignfold_hyperparams.param_basepair > 1.);
}
