echo "resnet_v2"
python di_ni_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/di_ni_fgsm_resnet_v2 &&
python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_resnet_v2 --source_model resnet_v2 &&
echo "inceptionv3"
python di_ni_fgsm.py  --source_model InceptionV3 --output_dir ./outputs/di_ni_fgsm_inceptionv3 &&
python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_inceptionv3 --source_model InceptionV3 &&
echo "InceptionV4"
python di_ni_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/di_ni_fgsm_inceptionv4 &&
python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_inceptionv4 --source_model InceptionV4 &&
echo "InceptionResnetV2"
python di_ni_fgsm.py  --source_model InceptionResnetV2 --output_dir ./outputs/di_ni_fgsm_InceptionResnetV2 &&
python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_InceptionResnetV2 --source_model InceptionResnetV2

# echo "resnet_v2"
# python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_resnet_v2 --source_model resnet_v2 &&
# echo "inceptionv3"
# python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_inceptionv3 --source_model InceptionV3 &&
# echo "InceptionV4"
# python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_inceptionv4 --source_model InceptionV4 &&
# echo "InceptionResnetV2"
# python simple_eval.py --EXP_ID di_ni_fgsm --input_dir ./outputs/di_ni_fgsm_InceptionResnetV2 --source_model InceptionResnetV2