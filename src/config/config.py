seq_len_tag = 'seq_lengths'
attention_type_sum = 'sum'
skip_frame_tag = 'skip_frame_ratio'

modality_mask_suffix_tag = '_mask'
modality_seq_len_tag='_seq_len'
modality_mask_tag = '_modality_mask'

inside_modality_tag = 'inside'
outside_modality_tag = 'outside'
gaze_modality_tag = 'gaze'
pose_modality_tag = 'pose'

activity_tag = 'activity'

dataset_split_tag = 'split'
train_dataset_tag = 'train'
test_dataset_tag = 'test'

# Labels in each task
# num_labels_in_task = {'Emotion':2,
#                     'Stress1':3,
#                     'Stress2':3,
#                     'Anxiety':2,
#                     'MentalWorkload':2}

num_labels_in_task = {'Emotion':4,
                    'Stress1':3,
                    'Stress2':3,
                    'Anxiety':2,
                    'MentalWorkload':2}

domain_labels = {'Stress1':1,
                 'Anxiety':2,
                 'MentalWorkload':3,
                 'Stress2':1,
                 'Emotion':4}

num_labels_in_task_mt = {'Emotion':4,
                    'Stress1':3,
                    'Stress2':3,
                    'Anxiety':2,
                    'MentalWorkload':2}


# Model tag
mm_attn_encoder = 'mm_attn_encoder'
hamlet_encoder = 'hamlet_encoder'
keyless_encoder = 'keyless_encoder'
gat_attn_encoder = 'gat_attn_encoder'
mammal_encoder = 'mammal_encoder'


#Logging config
tbw_train_loss = 'Loss/Train'
tbw_valid_loss = 'Loss/Valid'
tbw_test_loss = 'Loss/Test'

tbw_train_acc = 'Accuracy/Train'
tbw_valid_acc = 'Accuracy/Valid'
tbw_test_acc = 'Accuracy/Test'

tbw_train_f1 = 'F1/Train'
tbw_valid_f1 = 'F1/Valid'
tbw_test_f1 = 'F1/Test'

tbw_train_precision = 'Precision/Train'
tbw_valid_precision = 'Precision/Valid'
tbw_test_precision = 'Precision/Test'

tbw_train_recall = 'Recall/Train'
tbw_valid_recall = 'Recall/Valid'
tbw_test_recall = 'Recall/Test'


seq_len_tag = 'seq_lengths'
modality_mask_tag='_mask'
modality_seq_len_tag='_seq_len'

attention_type_sum = 'sum'
