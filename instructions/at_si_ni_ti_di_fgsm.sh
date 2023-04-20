echo "resnet_v2"
python at_si_ni_ti_di_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/at_si_ni_ti_di_fgsm_resnet_v2 --cuda_device 1 &&
python simple_eval.py --EXP_ID at_si_ni_ti_di_fgsm --input_dir ./outputs/at_si_ni_ti_di_fgsm_resnet_v2/adv-debug --source_model resnet_v2 --cuda_device 1 &&
echo "inceptionv3"
python at_si_ni_ti_di_fgsm.py   --output_dir ./outputs/at_si_ni_ti_di_fgsm_inceptionv3 --cuda_device 3&&
python simple_eval.py --EXP_ID at_si_ni_ti_di_fgsm --input_dir ./outputs/at_si_ni_ti_di_fgsm_inceptionv3/adv-debug --source_model InceptionV3 --cuda_device 1&&
echo "InceptionV4"
python at_si_ni_ti_di_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/at_si_ni_ti_di_inceptionv4 &&
python simple_eval.py --EXP_ID at_si_ni_ti_di_fgsm --input_dir ./outputs/si_ni_ti_di_inceptionv4/adv-debug --source_model InceptionV4 --cuda_device 2&&
echo "InceptionResnetV2"
python at_si_ni_ti_di_fgsm.py  --source_model InceptionResnetV2 --output_dir ./outputs/at_si_ni_ti_di_InceptionResnetV2 --cuda_device 0&&
python simple_eval.py --EXP_ID at_si_ni_ti_di_fgsm --input_dir ./outputs/at_si_ni_ti_di_InceptionResnetV2/adv-debug --source_model InceptionResnetV2 --cuda_device 3