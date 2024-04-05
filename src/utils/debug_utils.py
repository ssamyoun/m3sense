
from torch.utils.data import DataLoader
from src.config import config

def debug_dataloader(dataset, collate_fn, args, last_batch=-1):
    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True, 
                                  collate_fn=collate_fn,
                                  num_workers=args.num_workers)
    print(len(train_dataloader))
    for batch_id, batch in enumerate(train_dataloader, 0):
        # print('batch_id', batch_id)
        for modality in args.modalities:
            continue
            # print(f'{modality} size:', batch[modality].size())
            # print(f'{modality} mask size:', batch[modality + config.modality_mask_suffix_tag].size())
            # print(f'{modality} len', batch[modality + config.modality_seq_len_tag],
            #       batch[modality + config.modality_seq_len_tag].max())

            # print(modality, mask)
        # print(f'modality mask size:', batch['modality_mask'].size())
        # print(f'label size:', batch['label'].size())
        # print(batch["modality_mask"])
        # print(f'label:', batch['label'])

        # if (batch_id > last_batch) and last_batch!=-1:
        #     break