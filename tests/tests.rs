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
    let mut seq = convert(fasta_record.seq());
    seq.insert(0, PSEUDO_BASE);
    seq.push(PSEUDO_BASE);
    let seq_len = seq.len();
    if seq_len > max_seq_len {
      max_seq_len = seq_len;
    }
    fasta_records.push(FastaRecord::new(String::from(fasta_record.id()), seq));
  }
  let num_of_threads = num_cpus::get() as NumOfThreads;
  let mut thread_pool = Pool::new(num_of_threads);
  let output_dir_path = Path::new(OUTPUT_DIR_PATH);
  let (sa, feature_scores) = wrapped_consalign::<u8, u8>(
    &mut thread_pool,
    &fasta_records,
    output_dir_path,
    input_file_path,
    DEFAULT_MIN_BPP_ALIGN,
    DEFAULT_MIN_ALIGN_PROB_ALIGN,
    ScoringModel::Ensemble,
    TrainType::TrainedTransfer,
    false,
    DEFAULT_MIN_BPP_ALIGN_TURNER,
    DEFAULT_MIN_ALIGN_PROB_ALIGN_TURNER,
    false,
  );
  let num_of_rnas = sa.rna_ids.len();
  let pos_maps_with_gaps_only = vec![0; num_of_rnas];
  for (i, pos_maps) in sa.struct_align.seq_align.pos_map_sets.iter().enumerate() {
    assert!(*pos_maps != pos_maps_with_gaps_only);
    if i == 0 {continue;}
    let prev_pos_maps = &sa.struct_align.seq_align.pos_map_sets[i - 1];
    for (&pos, &prev_pos) in pos_maps.iter().zip(prev_pos_maps.iter()) {
      assert!(pos > prev_pos || pos == 0);
    }
  }
  assert!(feature_scores.align_count_posterior > 1. && feature_scores.basepair_count_posterior > 1.);
}
