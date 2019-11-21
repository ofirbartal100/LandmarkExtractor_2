import argparse
import os
import fnmatch
import pandas as pd
from tqdm import tqdm

# this function should change, depending on the dataset
def find_matching_label_ganarated_hands(file_path):
    labels_path = file_path.replace("color_composed.png", "joint2D.txt")
    with open(labels_path) as labels_file:
        labels = labels_file.read()
        return labels.split(",")


def find_recursive(root_dir, label_matching_methond, ext_list=['.png', '.jpg']):
    dataset = []
    counter = 0
    for root, dirnames, filenames in os.walk(root_dir):
        for ext in ext_list:
            for filename in fnmatch.filter(filenames, '*' + ext):
                file_path = os.path.join(root, filename)
                matching_label = label_matching_methond(file_path)
                dataset.append({"image": file_path, "label": matching_label})
        counter = counter + 1
        print(f"finished {counter} directories..")
    return dataset


def main(args):
    ds = find_recursive(args.imgs_dir,find_matching_label_ganarated_hands)

    # shuffle pdf
    df = pd.DataFrame(ds)
    train = df.sample(frac=(1-(args.test_ratio+args.val_ratio)), random_state=200)
    temp_df = df.drop(train.index)
    val = temp_df.sample(frac=(args.val_ratio/(args.test_ratio+args.val_ratio)), random_state=200)
    test = temp_df.drop(val.index)


    # save files to dir
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    train.to_csv(os.path.join(args.out_path,"train_dataset.csv"))
    val.to_csv(os.path.join(args.out_path,"val_dataset.csv"))
    test.to_csv(os.path.join(args.out_path,"test_dataset.csv"))
    print("done generating csv files!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate CSV For GanaratedHands Dataset"
    )
    parser.add_argument(
        "--imgs_dir",
        default="/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/data/noObject/",
        metavar="FOLDER",
        help="path to root images folder",
        type=str,
    )

    parser.add_argument(
        "--out_path",
        default="/disk1/ofirbartal/Projects/Dataset/GANeratedHands_Release/dataset_csv",
        metavar="FILE",
        help="path to output folder",
        type=str,
    )

    parser.add_argument(
        "--val_ratio",
        default=0.2,
        metavar="FLOAT",
        help="portion of the data that will be used as a validation set",
        type=float,
    )

    parser.add_argument(
        "--test_ratio",
        default=0.2,
        metavar="FLOAT",
        help="portion of the data that will be used as a test set",
        type=float,
    )

    args = parser.parse_args()

    main(args)
