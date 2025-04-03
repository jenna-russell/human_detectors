import openai
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
import re
import html2text

def model_call(prompt, model="gpt-4o-2024-08-06", temp=0.0):
    """
    API call to OpenAI; no fancy flags, just change the model name here
    """
    if 'o1' in model:
        completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        reasoning_effort='high'
        )
    else:             
        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp
        )

    return completion.choices[0].message.content

def myround(x, base):
    return base * round(x/base)

def create_prompt(title, subtitle, section, length, name_list=None, publication=""):
    rounded_len = myround(length, 50)
    if name_list:
        if publication == "":
            prompt = """You are given the following title and subtitle of a general article from the {} section and asked to write a corresponding article of around {} words. Include quotations from relevant experts and make sure the article is concise and easily understandable to a lay audience. 
    Title: {}
    Subtitle: {}
    {}""".format(section, rounded_len, title, subtitle, name_list)
        else:
            prompt = """You are given the following title and subtitle of a general article from the {} section of {} and asked to write a corresponding article of around {} words. Include quotations from relevant experts and make sure the article is concise and easily understandable to a lay audience.
    Title: {}
    Subtitle: {}
    {}""".format(section, publication, rounded_len, title, subtitle, name_list)
    else:
        if publication == "":
            prompt = """You are given the following title and subtitle of a general article from the {} section and asked to write a corresponding article of around {} words. Include quotations from relevant experts and make sure the article is concise and easily understandable to a lay audience. 
    Title: {}
    Subtitle: {}""".format(section, rounded_len, title, subtitle)
        else:
            prompt = """You are given the following title and subtitle of a general article from the {} section of {} and asked to write a corresponding article of around {} words. Include quotations from relevant experts and make sure the article is concise and easily understandable to a lay audience.
    Title: {}
    Subtitle: {}""".format(section, publication, rounded_len, title, subtitle)
    return prompt

def name_prompt(article):
    prompt = """Given the following article, please tell me the full names of every person mentioned in the article.
Article: {}""".format(article)
    return prompt

def num_tokens_from_string(string: str, encoding_name: str = "o200k_base") -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.
    Args:
    string (str): The text string to tokenize.
    encoding_name (str): The encoding name. Default is "o200k_base". New for gpt-4o.
    Returns:
    int: The number of tokens in the text string.
    """
    # Load the encoding
    encoding = tiktoken.get_encoding(encoding_name)
    # Convert the string into tokens and count them
    num_tokens = len(encoding.encode(string))
    return num_tokens

def distribution_stats(dat):
    print(f'Min: {np.min(dat)}')
    print(f'Q1: {np.quantile(dat, .25)}')
    print(f'Median: {np.median(dat)}')
    print(f'Mean: {np.mean(dat)}')
    print(f'Q3: {np.quantile(dat, .75)}')
    print(f'Max: {np.max(dat)}')

def clean_article_with_headings(title, subtitle, text):
    title = title.strip().lower()
    subtitle = subtitle.strip().lower()

    # Remove lines containing the title or subtitle
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if title not in line.lower() and subtitle not in line.lower()]
    
    # Convert bold, italics, strikethrough, and inline code
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)      # Bold alternative
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italics
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)        # Italics alternative
    text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)      # Strikethrough
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)  # Inline code
    
    # Convert links
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)  # Links

    # Convert Markdown headings into bold text
    text = re.sub(r'^\s*#{1,6}\s*(.*)', r'<b>\1</b>', text, flags=re.MULTILINE)
    
    # Remove leading/trailing whitespace and return cleaned text
    cleaned_text = "\n".join(cleaned_lines).strip()
    return cleaned_text

def html_to_markdown(html_content):
    """
    Converts HTML content to Markdown format.

    Args:
        html_content (str): The HTML string to convert.

    Returns:
        str: The converted Markdown string.
    """
    # Initialize the html2text converter
    converter = html2text.HTML2Text()
    # Configure the converter to ignore links and other options if needed
    converter.ignore_links = False
    converter.body_width = 0
    # Convert HTML to Markdown
    markdown_text = converter.handle(html_content)
    return markdown_text 