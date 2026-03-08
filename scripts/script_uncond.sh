python main.py --name etth_24 --config_file Config/uncond/etth.yaml --gpu 0 --mode uncond --model_name pacodi_sde --train
python main.py --name etth_24 --config_file Config/uncond/etth.yaml --gpu 0 --mode uncond --model_name pacodi_sde --milestone 10
python evaluate_uncond.py