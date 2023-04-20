echo "resnet_v2"
python bi_fgsm.py  --source_model resnet_v2 --output_dir ./outputs/bi_fgsm_resnet_v2 &&
python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_resnet_v2 --source_model resnet_v2 &&
echo "inceptionv3"
python bi_fgsm.py  --source_model InceptionV3 --output_dir ./outputs/bi_fgsm_inceptionv3 &&
python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_inceptionv3 --source_model InceptionV3 &&
python simple_eval.py --EXP_ID bi_fgsm_comdefend --input_dir /home/niujh/SI-NI-FGSM-master/defense/Comdefend/result_adv --source_model InceptionV3 

echo "InceptionV4"
python bi_fgsm.py  --source_model InceptionV4 --output_dir ./outputs/bi_fgsm_inceptionv4 &&
python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_inceptionv4 --source_model InceptionV4 &&
echo "InceptionResnetV2"
python bi_fgsm.py  --source_model InceptionResnetV2 --output_dir ./outputs/bi_fgsm_InceptionResnetV2 &&
python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_InceptionResnetV2 --source_model InceptionResnetV2 
/home/niujh/SI-NI-FGSM-master/defense/Comdefend/result_adv
#进行评估操作
# echo "resnet_v2"
# python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_resnet_v2 --source_model resnet_v2 &&
# echo "inceptionv3"
# python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_inceptionv3 --source_model InceptionV3 &&
# echo "InceptionV4"
# python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_inceptionv4 --source_model InceptionV4 &&
# echo "InceptionResnetV2"
# python simple_eval.py --EXP_ID bi_fgsm --input_dir ./outputs/bi_fgsm_InceptionResnetV2 --source_model InceptionResnetV2