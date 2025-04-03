import numpy as np 
import os
from bs4 import BeautifulSoup
import spacy
import re

nlp = spacy.load("en_core_web_sm")

base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
with open(os.path.join(base_dir, "prompts/paraphrase_prompts/paraphrase_sent_init.txt")) as f:
    paraphrase_template_init = f.read()
with open(os.path.join(base_dir, "prompts/paraphrase_prompts/paraphrase_sent.txt")) as f:
    paraphrase_template = f.read()

def paraphrase(text, paraphraser, publication="", section=""):
    doc = nlp(text)
    sentences = [str(s) for s in doc.sents]
    pp_sentences = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            prompt = paraphrase_template_init.format(section, publication, sentence)
            pp_sentence = paraphraser.generate(prompt, max_tokens=200, temperature=0)
            pp_sentences.append(pp_sentence)
            continue
        prompt = paraphrase_template.format(section, publication, ' '.join(pp_sentences), sentence)
        pp_sentence = paraphraser.generate(prompt, max_tokens=200, temperature=0)
        pp_sentences.append(pp_sentence)
    pp_text = ' '.join(pp_sentences)
    return pp_text

def paraphrase_html(text, paraphraser, publication="", section=""):
    soup = BeautifulSoup(text, "html.parser")
    segments = [segment.strip() for segment in soup.decode_contents().split('<br/>')]

    bold_texts = []
    pp_sentences = []
    first_sentence = True
    pp_texts = []

    for bold in soup.find_all('b'):
        bold_texts.append(str(bold))
        
    for segment in segments:
        pp_segment_sentences = []
        format = "normal"

        if not segment.strip():
            format = 'blank'
            pp_segment_sentences.append("<br/><br/>")
       
        if segment in bold_texts:
            format = "bolded"
            segment = segment.replace("<b>", "") 
            segment = segment.replace("</b>", "")

        doc = nlp(segment)
        segment_sentences = [str(s) for s in doc.sents]

        for i, sentence in enumerate(segment_sentences):
            if first_sentence:
                prompt = paraphrase_template_init.format(section, publication, sentence)
                if publication != "": 
                    prompt += " Please make the "
                pp_sentence = paraphraser.generate(prompt, max_tokens=200, temperature=0)
                pp_sentences.append(pp_sentence)
                pp_segment_sentences.append(pp_sentence)
                first_sentence = False
                continue
            prompt = paraphrase_template.format(section, publication, ' '.join(pp_sentences), sentence)
            pp_sentence = paraphraser.generate(prompt, max_tokens=200, temperature=0)
            pp_sentences.append(pp_sentence)
            pp_segment_sentences.append(pp_sentence)

        segment_text = ' '.join(pp_segment_sentences)
        if format == "bolded":
            segment_text = '<b>' + segment_text + '</b>'
        pp_texts.append(segment_text)

    pp_text = ' '.join(pp_texts)
    return pp_text, pp_texts

def print_tpr_target(fpr, tpr, target_fpr=0.01):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    if target_fpr < fpr[0]:
        print(f"TPR at {target_fpr*100}% FPR: {tpr[0] * 100:5.1f}% (target too low)")
        return
    
    if target_fpr > fpr[-1]:
        print(f"TPR at {target_fpr*100}% FPR: {tpr[-1] * 100:5.1f}% (target too high)")
        return
    
    idx = np.searchsorted(fpr, target_fpr, side='right')
    
    if fpr[idx-1] == target_fpr:
        tpr_value = tpr[idx-1]
    else:
        tpr_value = tpr[idx-1] + (target_fpr - fpr[idx-1]) * (tpr[idx] - tpr[idx-1]) / (fpr[idx] - fpr[idx-1])
    
    print(f"TPR at {target_fpr*100}% FPR: {tpr_value * 100:5.1f}%") 