# Human Detectors

[![arxiv](https://img.shields.io/badge/arXiv-2501.15654-b31b1b.svg)](http://arxiv.org/abs/2501.15654)



## DATA


Annotation data is stored in a JSON file, formatted as follows:

```markdown
* `generation_model`: (str) 
* `prompt_id`: (int)
* `title`: (str)
* `sub-title`: (str)
* `author`: (str)
* `source`: (str)
* `issue`: (str)
* `section`: (str)
* `link`: (str)
* `article`: (str)
* `id`: (int)
* `ground_truth`: (str)
* `pangram`: (str)
* `pangram_humanizers`: (str)
* `gptzero`: (str)
* `e5_lora`: (int)
* `RADAR`: (int)
* `annotator_1`: (dict)
      * `guess`: (str)
      * `confidence`: (int)
      * `comment`: (str)
* `annotator_2`: (dict)
      * `guess`: (str)
      * `confidence`: (int)
      * `comment`: (str)
* `annotator_3`: (dict)
      * `guess`: (str)
      * `confidence`: (int)
      * `comment`: (str)
* `annotator_4`: (dict)
      * `guess`: (str)
      * `confidence`: (int)
      * `comment`: (str)
* `annotator_5`: (dict)
      * `guess`: (str)
      * `confidence`: (int)
      * `comment`: (str)
```

Example entry
```json
{"0":
   {
    "generation_model":"gpt-4o",
    "prompt_id":1,
    "title":"Louisiana schools won't display Ten Commandments before November as lawsuit plays out",
    "sub-title":"Louisiana has agreed to delay implementing a requirement that the Ten Commandments be placed in all of the state\u2019s public school classrooms, at least until November.",
    "author":"Kevin McGill",
    "source":"Associated Press",
    "issue":"7/19/24",
    "section":"Education",
    "link":"https://apnews.com/search?q=Louisiana+schools+won%E2%80%99t+display+Ten+Commandments+before+November+as+lawsuit+plays+out#nt=navsearch",
    "article":"Louisiana won't take official steps to implement a law requiring the Ten ...",
    "id":1,
    "ground_truth":"Human-written",
    "pangram":"pangram_output",
    "pangram_humanizers":"pangram_humanizers_output",
    "gptzero":"gptzero_output",
    "e5_lora":0.0420721397,
    "RADAR":0.8631911874,
    "annotator_1":
      {
       "guess":"Human-Generated",
       "confidence":4,
       "comment":"The article overall is very factual and looks quite well-researched. It reads like a standard news story..."
      },
    "annotator_2":
      {
       "guess":"Human-Generated",
       "confidence":4,
       "comment":"I don't see any of the typical words used by AI. Also, the sentences are longer and more complex than..."
      },
    "annotator_3":
      {
       "guess":"Human-Generated",
       "confidence":4,
       "comment":"Some of the phrasing sounds slightly awkward (highlighted), and there're places where the punctuation is off."},
    "annotator_4":
      {
       "guess":"Human-Generated",
       "confidence":5,
       "comment":"The article appears human-written. While many of the sentences are long, they're packed with information..."},
    "annotator_5":
      {
       "guess":"Human-Generated",
       "confidence":5,
       "comment":"Although the author tries to report only facts, the final sentence..."
      }
   },
}

```




## Citation Information
If you use this dataset, please cite it as follows:
```
@misc{russell-et-al-2025,
      title={People who frequently use ChatGPT for writing tasks are accurate and robust detectors of AI-generated text}, 
      author={Jenna Russell and Marzena Karpinska and Mohit Iyyer},
      year={2025},
      eprint={2501.15654},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
