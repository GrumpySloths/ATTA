echo "resnet_v2"
python si_ni_ti_di_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/si_ni_ti_di_fgsm_resnet_v2 --cuda_device 1 &&
python simple_eval.py --EXP_ID si_ni_ti_di_fgsm --input_dir ./outputs/si_ni_ti_di_fgsm_resnet_v2 --source_model resnet_v2 --cuda_device 1 &&
echo "inceptionv3"
python si_ni_ti_di_fgsm.py   --output_dir ./outputs/si_ni_ti_di_fgsm_inceptionv3 &&
python simple_eval.py --EXP_ID si_ni_ti_di_fgsm --input_dir ./outputs/si_ni_ti_di_fgsm_inceptionv3 --source_model InceptionV3 &&
echo "InceptionV4"
python si_ni_ti_di_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/si_ni_ti_di_inceptionv4 &&
python simple_eval.py --EXP_ID si_ni_ti_di_fgsm --input_dir ./outputs/si_ni_ti_di_inceptionv4 --source_model InceptionV4 --cuda_device 2&&
echo "InceptionResnetV2"
python si_ni_ti_di_fgsm.py  --source_model InceptionResnetV2 --output_dir ./outputs/si_ni_ti_di_InceptionResnetV2 --cuda_device 0&&
python simple_eval.py --EXP_ID si_ni_ti_di_fgsm --input_dir ./outputs/si_ni_ti_di_InceptionResnetV2 --source_model InceptionResnetV2 --cuda_device 3