python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset county_facebook_2016 --target income
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset county_facebook_2016 --target education
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset county_facebook_2016 --target unemployment
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset county_facebook_2016 --target election

python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset climate_2008         --target airT
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset climate_2008         --target landT
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset climate_2008         --target precipitation
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset climate_2008         --target sunlight
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset climate_2008         --target pm2.5

python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset ward_2016            --target income
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset ward_2016            --target edu
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset ward_2016            --target age
python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset ward_2016            --target election

python train_model.py --ppnp PPNP     --RPMAX 10 --net APPNP  --dataset twitch_PTBR_true     --target days



python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset county_facebook_2016 --target income
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset county_facebook_2016 --target education
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset county_facebook_2016 --target unemployment
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset county_facebook_2016 --target election

python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset climate_2008         --target airT
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset climate_2008         --target landT
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset climate_2008         --target precipitation
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset climate_2008         --target sunlight
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset climate_2008         --target pm2.5

python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset ward_2016            --target income
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset ward_2016            --target edu
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset ward_2016            --target age
python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset ward_2016            --target election

python train_model.py --ppnp GPR_prop --RPMAX 10 --net GPRGNN --dataset twitch_PTBR_true     --target days
