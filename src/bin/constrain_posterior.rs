extern crate consalign;

use consalign::*;
use std::env;

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_dir",
    "A path to an input directory containing RNA structural alignments in STOCKHOLM format",
    "STR",
  );
  opts.reqopt(
    "o",
    "output_file_path",
    "A path to an output file to record intermediate training log likelihoods",
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
  opts.optopt("", "mix_weight", &format!("A mixture weight (Uses {} by default)", DEFAULT_MIX_WEIGHT), "FLOAT");
  opts.optopt("t", "num_of_threads", "The number of threads in multithreading (Uses the number of the threads of this computer by default)", "UINT");
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
  let input_dir_path = matches.opt_str("i").unwrap();
  let input_dir_path = Path::new(&input_dir_path);
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
  } as u16;
  let mix_weight = if matches.opt_present("mix_weight") {
    matches.opt_str("mix_weight").unwrap().parse().unwrap()
  } else {
    DEFAULT_MIX_WEIGHT
  };
  let num_of_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumOfThreads
  };
  println!("# threads = {}", num_of_threads);
  let output_file_path = matches.opt_str("o").unwrap();
  let output_file_path = Path::new(&output_file_path);
  let entries: Vec<DirEntry> = read_dir(input_dir_path).unwrap().map(|x| x.unwrap()).filter(|x| match x.path().extension().unwrap().to_str().unwrap() {"sto" | "stk" | "sth" => true, _ => false,}).collect();
  let num_of_entries = entries.len();
  let mut train_data = vec![TrainDatumPosterior::origin(); num_of_entries];
  let mut thread_pool = Pool::new(num_of_threads);
  for (input_file_path, train_datum) in entries.iter().zip(train_data.iter_mut()) {
    *train_datum = TrainDatumPosterior::<u16>::new(&input_file_path.path(), min_bpp, offset_4_max_gap_num, &mut thread_pool, mix_weight);
  }
  constrain_posterior::<u16>(&mut thread_pool, &mut train_data, output_file_path);
}
