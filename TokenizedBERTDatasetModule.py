import bisect
import random
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer
from typing import Iterable

class TokenizedBERTDataset(ConcatDataset):
    #OBS: The 'step_counter' variable counts the number of times a value is etracted from the Dataset, i.e. the number of times that __getitem__ is called.

    @property
    def features(self):
        features = ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'next_sentence_label']
        if(self.include_raw_text_flag):
            features.append('text')
        if(self.include_idx_flag):
            features.append('idx')
        return features

    @property
    def sequence_length_info(self):
        if(self.sequence_length_changes is not None):
            return {
                "starting_max_sequence_length": self.max_sequence_length,
                "sequence_length_changes": self.sequence_length_changes,
                "sequence_length_changes_step": self.sequence_length_changes_step,
                "longest_sequence_length": self.longest_sequence_length
            }
        else:
            return {"max_sequence_length": self.max_sequence_length}
    
    def __init__(self, datasets: Iterable[Dataset], tokenizer_checkpoint='bert-base-cased', max_sequence_length = 512, sequence_length_changes=None,
                 sequence_length_changes_step=None, truncate_resulting_item_flag=False, include_idx_flag=False, include_raw_text_flag=False,
                 print_iteration_info_flag=False, print_randomness_info_flag=False):
        super().__init__(datasets)
        self.step_counter = 0
        self.tokenizer_vocab = AutoTokenizer.from_pretrained(tokenizer_checkpoint).vocab
        self.max_sequence_length = max_sequence_length
        self.sequence_length_changes = sequence_length_changes
        self.sequence_length_changes_step = sequence_length_changes_step
        self.include_idx_flag = include_idx_flag
        self.include_raw_text_flag = include_raw_text_flag
        self.truncate_resulting_item_flag = truncate_resulting_item_flag
        self.print_iteration_info_flag = print_iteration_info_flag
        self.print_randomness_info_flag = print_randomness_info_flag
        self.__determine_longest_sequence_length__()

    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __repr__(self):
        return f"Dataset({{\n    features: {self.features},\n    num_entries: {self.cumulative_sizes[-1]}\n}})"
    
    def __apply_mask__(self, token_sequence):
        non_masked_token_label = -100
        masked_sequence = []
        label_sequence = []
        for i, token in enumerate(token_sequence):
            if(token != 101 and token != 102):
                probability = random.random()
    
                if(probability <= 0.15):
                    probability /= 0.15
    
                    if(probability <= 0.8):
                        masked_sequence.append(103)
                    elif(probability <= 0.9):
                        masked_sequence.append(random.randrange(len(self.tokenizer_vocab)))
                    else:
                        masked_sequence.append(token)
    
                    label_sequence.append(token)
                    continue
            
            masked_sequence.append(token)
            label_sequence.append(non_masked_token_label)
        return masked_sequence, label_sequence

    def __build_sentence_pair__(self, idx):
        first_raw_item = self.__get_raw_item__(idx)
        second_raw_item = {'idx': -1, 'text': ""}
        next_sentence_label = 0
        random_result = random.random()
        if(self.print_randomness_info_flag):
            print(random_result)
        if(random_result < 0.5):
            random_index = -1
            while random_index < 0 or random_index == idx:
                random_index = random.randrange(0, self.cumulative_sizes[-1])
            second_raw_item = self.__get_raw_item__(random_index)
            next_sentence_label = 1
        elif(idx < self.cumulative_sizes[-1] - 1):
            second_raw_item = self.__get_raw_item__(idx + 1)
        return first_raw_item, second_raw_item, next_sentence_label

    def __combine_raw_items__(self, first_raw_item_text, second_raw_item_text):
        input_ids = [101] + first_raw_item_text + [102] + second_raw_item_text + [102]
        return {
            "input_ids": input_ids,
            "token_type_ids": [0 for _ in range(len(first_raw_item_text) + 2)] + [1 for _ in range(len(second_raw_item_text) + 1)],
            "attention_mask": [1 for _ in range(len(input_ids))]
        }

    def __determine_longest_sequence_length__(self):
        self.longest_sequence_length = self.max_sequence_length
        if(self.sequence_length_changes is not None):
            for length_change in self.sequence_length_changes:
                if(length_change > self.longest_sequence_length):
                    self.longest_sequence_length = length_change
    
    def __increment_idx__(self, idx):
        new_idx = idx + 1
        if(new_idx >= self.cumulative_sizes[-1]):
            new_idx = 0
        return new_idx
    
    def __get_raw_item__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        raw_item = self.datasets[dataset_idx][sample_idx]
        raw_item['idx'] = idx
        return raw_item

    def __pad_item__(self, item):
        if(len(item['input_ids']) < self.longest_sequence_length):
            padding_length = self.longest_sequence_length - len(item['input_ids'])
            for _ in range(0, padding_length):
                item['input_ids'].append(0)
                item['token_type_ids'].append(0)
                item['attention_mask'].append(0)
                item['labels'].append(-100)
        return item

    def __truncate_item__(self, item):
        return {
            "input_ids": item['input_ids'][:(self.max_sequence_length - 1)] + [102],
            "token_type_ids": item['token_type_ids'][:self.max_sequence_length],
            "attention_mask": item['attention_mask'][:self.max_sequence_length]
        }
    
    def __update_max_sequence_length__(self):
        if(self.step_counter >= self.sequence_length_changes_step[0]):
            self.max_sequence_length = self.sequence_length_changes[0]
            del self.sequence_length_changes[0]
            del self.sequence_length_changes_step[0]
            if(len(self.sequence_length_changes) == 0):
                self.sequence_length_changes = None
                self.sequence_length_changes_step = None
    
    def __getitem__(self, idx):
        self.step_counter += 1
        if(self.sequence_length_changes is not None):
            self.__update_max_sequence_length__()
        if(self.print_iteration_info_flag):
            print("\nCurrent Step: ", self.step_counter)
            print("Starting Index: ", idx)
        item = {}
        while True:
            first_raw_item, second_raw_item, next_sentence_label = self.__build_sentence_pair__(idx)
            item = self.__combine_raw_items__(first_raw_item['text'], second_raw_item['text'])
            if(self.print_iteration_info_flag):
                print("Current Sequence Length: ", len(item['input_ids']))
                print("Maximum Sequence Length: ", self.max_sequence_length)
            if(len(item['input_ids']) > self.max_sequence_length):
                if(not self.truncate_resulting_item_flag):
                    idx = self.__increment_idx__(idx)
                    if(self.print_iteration_info_flag):
                        print("Incremented Index: ", idx)
                    continue
                else:
                    item = self.__truncate_item__(item)
            item['input_ids'], item['labels'] = self.__apply_mask__(item['input_ids'])
            if(self.include_idx_flag):
                item['idx'] = [first_raw_item['idx'], second_raw_item['idx']]
            item['next_sentence_label'] = next_sentence_label
            if(self.include_raw_text_flag):
                item['text'] = [first_raw_item['text'], second_raw_item['text']]
            item = self.__pad_item__(item)
            break
        return item
