import os,argparse
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('--origin', type=dir_path)
args = parser.parse_args()

origin = args.origin
## Etape 2 : Copy ground_truth & detection_results txt files to mAP folder :
os.system('rm ./mAP/input/detection-results/*')
os.system('rm ./mAP/input/ground-truth/*')
os.system('rm ./mAP/input/images/*')
os.system('rm -r ./mAP/output')
print("./{}/Prediction/*".format(origin))
os.system('cp -R ./{}/Prediction/* ./mAP/input/detection-results/'.format(origin))
os.system('cp -R ./{}/*.txt ./mAP/input/ground-truth/'.format(origin))
os.system('cp -R ./{}/*.jpeg ./mAP/input/images/'.format(origin))
## Etape 3 : Convert to convenient format using the python scripts :
os.system('python3 ./mAP/scripts/extra/convert_dr_perso_yolo.py')
os.system('python3 ./mAP/scripts/extra/convert_gt_yolo.py')
## Etape 4 : Compute mAP :
os.system('python3 ./mAP/scripts/extra/intersect-gt-and-dr.py') # optional if there is an error to be avoided
os.system('python3 ./mAP/main.py --no-plot')