echo "resnet_v2"
python si_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/si_fgsm_resnet_v2 --cuda_device 0 &&
python simple_eval.py --EXP_ID si_fgsm --input_dir ./outputs/si_fgsm_resnet_v2 --source_model resnet_v2 --cuda_device 0 &&
echo "inceptionv3"
python si_fgsm.py   --output_dir ./outputs/si_fgsm_inceptionv3 --cuda_device 1&&
python simple_eval.py --EXP_ID si_fgsm --input_dir ./outputs/si_fgsm_inceptionv3  &&
echo "InceptionV4"
python si_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/si_fgsm_inceptionv4 --cuda_device 1&&
python simple_eval.py --EXP_ID si_fgsm --input_dir ./outputs/si_fgsm_inceptionv4 --source_model InceptionV4 --cuda_device 1&&
echo "InceptionResnetV2"
python si_fgsm.py  --source_model InceptionResnetV2 --output_dir ./outputs/si_fgsm_InceptionResnetV2 --cuda_device 2&&
python simple_eval.py --EXP_ID si_fgsm --input_dir ./outputs/si_fgsm_InceptionResnetV2 --source_model InceptionResnetV2 --cuda_device 2