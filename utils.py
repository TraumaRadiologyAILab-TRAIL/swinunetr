import os
import pandas as pd
def split_data(training_folds=['0', '1', '2'], val_folds=['3'], test_fold=['4'], series_dict=None, image_dict=None):
        train_df = pd.DataFrame(columns=['case_id', 'htx','mh','case_path','mask_path'])
        val_df = pd.DataFrame(columns=['case_id', 'htx','mh','case_path','mask_path'])
        test_df = pd.DataFrame(columns=['case_id', 'htx','mh','case_path','mask_path'])
        case_base_path = '/storage/data/nnunet/nnUNet_Plans_3d_fullres'
        print(f"Total cases: {len(series_dict)}")
        print(f'all data: {len(os.listdir(case_base_path))}')
        for case_id, value in series_dict.items():
            if value['fold'] in training_folds:
                target = 'train'
            elif value['fold'] in val_folds:
                target = 'val'
            elif value['fold'] in test_fold:
                target = 'test'
            else:
                continue
            
            case_path = os.path.join(case_base_path, case_id + '.npz')
            if os.path.exists(case_path):
                if target == 'train':
                    train_df = train_df._append({'case_id': case_id, 'htx': value['htx'], 'mh': value['mh'], 'case_path': case_path}, ignore_index=True)
                elif target == 'val':
                    val_df = val_df._append({'case_id': case_id, 'htx': value['htx'], 'mh': value['mh'], 'case_path': case_path}, ignore_index=True)
                elif target == 'test':
                    test_df = test_df._append({'case_id': case_id, 'htx': value['htx'], 'mh': value['mh'], 'case_path': case_path}, ignore_index=True)
    
        return train_df, val_df, test_df