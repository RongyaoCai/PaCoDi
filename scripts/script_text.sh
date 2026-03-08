python main.py --name etth1_24 --config_file Config/text/etth1.yaml --gpu 0 --mode text --model_name pacodi_ddpm --train
python main.py --name etth1_24 --config_file Config/text/etth1.yaml --gpu 0 --mode text --model_name pacodi_ddpm --milestone 10 --num_samples 5
python evaluate_text.py