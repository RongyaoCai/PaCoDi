python main.py --name etth_24 --config_file Config/uncond/etth.yaml --gpu 0 --mode uncond --model_name pacodi --train
python main.py --name etth_24 --config_file Config/uncond/etth.yaml --gpu 0 --mode uncond --model_name pacodi --milestone 10
python evaluate/evaluate_uncond.py