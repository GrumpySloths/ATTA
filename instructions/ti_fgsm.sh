echo "resnet_v2"
python ti_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/ti_fgsm_resnet_v2 --cuda_device 3 &&
python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_resnet_v2 --source_model resnet_v2 &&
echo "inceptionv3"
python ti_fgsm.py  --source_model InceptionV3 --output_dir ./outputs/ti_fgsm_inceptionv3 --cuda_device 1&&
python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_inceptionv3 --source_model InceptionV3 --cuda_device 1&&
echo "InceptionV4"
python ti_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/ti_fgsm_inceptionv4 --cuda_device 2&&
python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_inceptionv4 --source_model InceptionV4 --cuda_device 2&&
echo "InceptionResnetV2"
python ti_fgsm.py  --source_model InceptionResnetV2 --output_dir ./outputs/ti_fgsm_InceptionResnetV2 --cuda_device 3&&
python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_InceptionResnetV2 --source_model InceptionResnetV2 --cuda_device 3

# echo "resnet_v2"
# python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_resnet_v2 --source_model resnet_v2 &&
# echo "inceptionv3"
# python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_inceptionv3 --source_model InceptionV3 &&
# echo "InceptionV4"
# python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_inceptionv4 --source_model InceptionV4 &&
# echo "InceptionResnetV2"
# python simple_eval.py --EXP_ID ti_fgsm --input_dir ./outputs/ti_fgsm_InceptionResnetV2 --source_model InceptionResnetV2