echo "resnet_v2"
python di_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/di_fgsm_resnet_v2 --cuda_device 3&&
python simple_eval.py --EXP_ID di_fgsm --input_dir ./outputs/di_fgsm_resnet_v2 --source_model resnet_v2 --cuda_device 3&&
echo "inceptionv3"
python di_fgsm.py  --source_model InceptionV3 --output_dir ./outputs/di_fgsm_inceptionv3 --cuda_device 2&&
python simple_eval.py --EXP_ID di_fgsm --input_dir ./outputs/di_fgsm_inceptionv3 --source_model InceptionV3 --cuda_device 2&&
echo "InceptionV4"
python di_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/di_fgsm_inceptionv4 --cuda_device 1&&
python simple_eval.py --EXP_ID di_fgsm --input_dir ./outputs/di_fgsm_inceptionv4 --source_model InceptionV4 --cuda_device 1&&
echo "InceptionResnetV2"
python di_fgsm.py  --source_model InceptionResnetV2 --output_dir ./outputs/di_fgsm_InceptionResnetV2 --cuda_device 0&&
python simple_eval.py --EXP_ID di_fgsm --input_dir ./outputs/di_fgsm_InceptionResnetV2 --source_model InceptionResnetV2 --cuda_device 0