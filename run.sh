# NON10
#python main.py --dataset="NON10" --lr=5e-4 --epochs=65 --detect_per_epoch=10 --fill_data_type="mean"
python main.py --dataset="NON10" --lr=5e-4 --epochs=65 --detect_per_epoch=10 --fill_data_type="season_mean_4" # 1
#python main.py --dataset="NON10" --lr=5e-4 --epochs=65 --detect_per_epoch=10 --fill_data_type="lowess"

# NON1388
#python main.py --fill_data_type="mean"
python main.py --fill_data_type="season_mean_4"
#python main.py --fill_data_type="lowess"

# SP500N
python main.py --dataset="SP500N" --lr=1e-4 --epochs=75 --detect_per_epoch=10 --fill_data_type="mean"
#python main.py --dataset="SP500N" --lr=1e-4 --epochs=75 --detect_per_epoch=10 --fill_data_type="season_mean_4"
#python main.py --dataset="SP500N" --lr=1e-4 --epochs=75 --detect_per_epoch=10 --fill_data_type="lowess"