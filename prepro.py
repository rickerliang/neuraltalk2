"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
--图片数 * rgb * 256 * 256
/labels is (M,max_length) uint32 array of encoded labels, zero padded
--每幅图片可能有若干个句子描述，若使用S1表示图片1对应的描述句子个数，那么
M = sigma(p=1->N)(Sp)
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
--标记每幅图片对应的描述句子在M的范围
/label_length stores the length of the sequence for each of the M sequences
--M内每个句子的长度

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize

def prepro_captions(imgs):
  
  # preprocess all the captions
  print 'example processed tokens:'
  for i,img in enumerate(imgs):
    img['processed_tokens'] = []
    for j,s in enumerate(img['captions']): # img['captions']是字符串数组 [s,s,s,s]
      # 去标点，分词(提取token)
      txt = str(s).lower().translate(None, string.punctuation).strip().split()
      # img['processed_tokens']是一个list[list]例如[[w,w,w,w],[w,w,w]]
      img['processed_tokens'].append(txt)
      if i < 10 and j == 0: print txt

def build_vocab(imgs, params):
  count_thr = params['word_count_threshold']

  # count up the number of words
  # map(token-count)
  counts = {}
  for img in imgs:
    for txt in img['processed_tokens']:
      for w in txt:
        counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
  print 'top words and their counts:'
  print '\n'.join(map(str,cw[:20]))

  # print some stats
  total_words = sum(counts.itervalues())
  print 'total words:', total_words
  bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
  vocab = [w for w,n in counts.iteritems() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
  print 'number of words in vocab would be %d' % (len(vocab), )
  print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

  # lets look at the distribution of lengths as well
  # map(sentenc_len-occur_times)
  sent_lengths = {}
  for img in imgs:
    for txt in img['processed_tokens']:
      nw = len(txt)
      sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print 'max length sentence in raw data: ', max_len
  print 'sentence length distribution (count, number of words):'
  sum_len = sum(sent_lengths.values())
  for i in xrange(max_len+1):
    print '%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
  # 针对img['processed_tokens']，小于threshold的token转成UNK token，保存到img['final_captions']
  # 所以img['final_captions']也是一个list[list]例如[[UNK,w,w,w],[w,UNK,w]]
  for img in imgs:
    img['final_captions'] = []
    for txt in img['processed_tokens']:
      caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
      img['final_captions'].append(caption)

  # 返回数量大于threshold的token list
  return vocab

def assign_splits(imgs, params):
  num_val = params['num_val']
  num_test = params['num_test']

  for i,img in enumerate(imgs):
      if i < num_val:
        img['split'] = 'val'
      elif i < num_val + num_test: 
        img['split'] = 'test'
      else: 
        img['split'] = 'train'

  print 'assigned %d to val, %d to test.' % (num_val, num_test)

def encode_captions(imgs, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  N = len(imgs)
  # 每幅图片可能有若干个句子描述，若使用S1表示图片1对应的描述句子个数，那么
  # M = sigma(p=1->N)(Sp)
  # 注意，img['final_captions']是一个list[list]
  M = sum(len(img['final_captions']) for img in imgs) # total number of captions

  label_arrays = []
  label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
  label_end_ix = np.zeros(N, dtype='uint32')
  label_length = np.zeros(M, dtype='uint32')
  caption_counter = 0
  counter = 1
  for i,img in enumerate(imgs):
    n = len(img['final_captions'])
    # n是每张图片对应的描述句子的个数
    assert n > 0, 'error: some image has no captions'
    
    Li = np.zeros((n, max_length), dtype='uint32')
    # Li是n行max_length列数组，Li[a][b]表示对应图片的第a个描述句子第b个词在vocab内的索引
    for j,s in enumerate(img['final_captions']):
      # s图片对应的描述句子
      label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
      caption_counter += 1
      for k,w in enumerate(s):
        if k < max_length:
          Li[j,k] = wtoi[w]

    # note: word indices are 1-indexed, and captions are padded with zeros
    # label_arrays是li的list
    # 例如
    # [[[w,w,w],[w,w,w]],
    #  [[w,w,w],[w,w,w]]]，label_array[i][j][k]对应第i张图片第j个句子第k个词在vocab的索引
    label_arrays.append(Li)
    label_start_ix[i] = counter
    label_end_ix[i] = counter + n - 1
    
    counter += n
  
  L = np.concatenate(label_arrays, axis=0) # put all the labels together
  # 所以L就是[[w,w,w],[w,w,w],[w,w,w],[w,w,w]]
  assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
  assert np.all(label_length > 0), 'error: some caption had no words?'

  print 'encoded captions to array of size ', `L.shape`
  return L, label_start_ix, label_end_ix, label_length

def main(params):

  imgs = json.load(open(params['input_json'], 'r'))
  seed(123) # make reproducible
  shuffle(imgs) # shuffle the order

  # tokenization and preprocessing
  # 去标点，分词(提取token)，保存到对应图片的['processed_tokens']，例如image1['processed_tokens']
  # img['processed_tokens']是一个list[list]例如[[w,w,w,w],[w,w,w]]
  prepro_captions(imgs)

  # create the vocab
  # 针对img['processed_tokens']，小于threshold的token转成UNK token，img的token经UNK转换后(如果有)保存到img['final_captions']
  # 所以img['final_captions']也是一个list[list]例如[[UNK,w,w,w],[w,UNK,w]]
  # 返回数量大于threshold的token list
  vocab = build_vocab(imgs, params)
  # map(index-token)
  itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  # map(token-index)
  wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

  # assign the splits
  # split train_set test_set validation_set
  assign_splits(imgs, params)
  
  # encode captions in large arrays, ready to ship to hdf5 file
  # 所以第i张图片第j个句子第k个词在vocab的索引表示为
  # L[label_start_ix[i] + j][k]
  L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

  # create output h5 file
  N = len(imgs)
  f = h5py.File(params['output_h5'], "w")
  f.create_dataset("labels", dtype='uint32', data=L)
  f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
  f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
  f.create_dataset("label_length", dtype='uint32', data=label_length)
  dset = f.create_dataset("images", (N,3,256,256), dtype='uint8') # space for resized images
  for i,img in enumerate(imgs):
    # load the image
    I = imread(os.path.join(params['images_root'], img['file_path']))
    try:
        Ir = imresize(I, (256,256))
    except:
        print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
        raise
    # handle grayscale input images
    if len(Ir.shape) == 2:
      Ir = Ir[:,:,np.newaxis]
      Ir = np.concatenate((Ir,Ir,Ir), axis=2)
    # and swap order of axes from (256,256,3) to (3,256,256)
    Ir = Ir.transpose(2,0,1)
    # write to h5
    dset[i] = Ir
    if i % 1000 == 0:
      print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)
  f.close()
  print 'wrote ', params['output_h5']

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['images'] = []
  for i,img in enumerate(imgs):
    
    jimg = {}
    jimg['split'] = img['split']
    if 'file_path' in img: jimg['file_path'] = img['file_path'] # copy it over, might need
    if 'id' in img: jimg['id'] = img['id'] # copy over & mantain an id, if present (e.g. coco ids, useful)
    
    out['images'].append(jimg)
  
  json.dump(out, open(params['output_json'], 'w'))
  print 'wrote ', params['output_json']

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--num_val', required=True, type=int, help='number of images to assign to validation data (for CV etc)')
  parser.add_argument('--output_json', default='data.json', help='output json file')
  parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
  
  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)
