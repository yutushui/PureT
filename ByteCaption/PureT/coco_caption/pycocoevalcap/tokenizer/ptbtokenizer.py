#!/usr/bin/env python
# 
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import tempfile
import itertools
import re

# path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = 'stanford-corenlp-3.4.1.jar'

# punctuations to be removed from the sentences
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
        ".", "?", "!", ",", ":", "-", "--", "...", ";"]

# Check if Java is available
_JAVA_AVAILABLE = False
_JAVA_PATH = 'java'
try:
    # Check for portable Java first
    portable_java = os.path.expanduser('~/java/jdk-11.0.2/bin/java')
    if os.path.exists(portable_java):
        _JAVA_PATH = portable_java
        _JAVA_AVAILABLE = True
    else:
        # Fall back to system Java
        result = subprocess.run(['which', 'java'], capture_output=True, text=True)
        if result.returncode == 0:
            _JAVA_AVAILABLE = True
except:
    pass

class PTBTokenizer:
    """Python wrapper of Stanford PTBTokenizer"""

    def _tokenize_python(self, captions_for_image):
        """Simple Python tokenization fallback when Java is not available."""
        final_tokenized_captions_for_image = {}
        for img_id, caps in captions_for_image.items():
            tokenized_caps = []
            for cap in caps:
                # Convert to lowercase
                caption = cap['caption'].lower() if isinstance(cap, dict) else cap.lower()
                # Remove punctuation
                for punct in PUNCTUATIONS:
                    caption = caption.replace(punct, '')
                # Simple word tokenization (split on whitespace)
                tokens = caption.split()
                # Remove empty tokens
                tokens = [t for t in tokens if t]
                tokenized_caption = ' '.join(tokens)
                tokenized_caps.append(tokenized_caption)
            final_tokenized_captions_for_image[img_id] = tokenized_caps
        return final_tokenized_captions_for_image

    def tokenize(self, captions_for_image):
        # Fallback to Python-only tokenization if Java is not available
        if not _JAVA_AVAILABLE:
            print("[PTBTokenizer] Java not available, using Python-only tokenization")
            return self._tokenize_python(captions_for_image)
        cmd = [_JAVA_PATH, '-cp', STANFORD_CORENLP_3_4_1_JAR, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        # ======================================================
        # prepare data for PTB Tokenizer
        # ======================================================
        final_tokenized_captions_for_image = {}
        image_id = [k for k, v in list(captions_for_image.items()) for _ in range(len(v))]
        sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in list(captions_for_image.items()) for c in v])

        # ======================================================
        # save sentences to temporary file
        # ======================================================
        path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
        tmp_root = os.environ.get("PTB_TOKENIZER_TMPDIR") or os.environ.get("TMPDIR") or path_to_jar_dirname
        if not os.path.isdir(tmp_root):
            tmp_root = path_to_jar_dirname
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=tmp_root)
        except OSError:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode('utf-8'))
        tmp_file.close()

        # ======================================================
        # tokenize sentence
        # ======================================================
        cmd.append(tmp_file.name)
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE)
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        lines = token_lines.decode("utf-8").split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        # ======================================================
        # create dictionary for tokenized captions
        # ======================================================
        for k, line in zip(image_id, lines):
            if not k in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                    if w not in PUNCTUATIONS])
            final_tokenized_captions_for_image[k].append(tokenized_caption)

        return final_tokenized_captions_for_image
