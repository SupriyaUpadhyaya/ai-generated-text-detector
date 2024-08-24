import sys, unicodedata
import os
import json
import re
from pylatexenc.latex2text import LatexNodes2Text

class DataPreprocessing:
    @staticmethod
    def convert_latex_to_text(latex_str):
        # Create a LatexNodes2Text object
        latex_to_text_converter = LatexNodes2Text()
        # Convert LaTeX string to plain text
        return latex_to_text_converter.latex_to_text(latex_str)

    @staticmethod
    def remove_inline_latex(text):
        # Regular expression pattern to match inline LaTeX expressions enclosed in $
        pattern = r'\$.*?\$'
        
        # Use re.sub() to replace the LaTeX expressions with an empty string
        cleaned_text = re.sub(pattern, '', text)
        
        return cleaned_text

    # function to remove non ascii
    @staticmethod
    def remove_non_ascii(text):
        try:
            print("before :", text)
            # Remove Unicode escape sequences
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            print("after :", text)
            return text
        except (UnicodeDecodeError, AttributeError, ValueError):
            return text
    
    
    @staticmethod
    def remove_non_printable(text):
        # Get all unicode characters
        all_chars = (chr(i) for i in range(sys.maxunicode))
        # Get all non printable characters
        control_chars = ''.join(c for c in all_chars if unicodedata.category(c) == 'Cc')
        # Create regex of above characters
        control_char_re = re.compile('[%s]' % re.escape(control_chars))
        # remove non-printable characters
        text = control_char_re.sub('', text)
        return text
    
    @staticmethod
    def remove_spaces(text):
        text = text.replace("\t", "").replace("\r", "").replace("\n", "").replace('\"', "'")     # remove \t, \n, \r
        text = re.sub('\s{2,}', '', text)    # remove 2 or more than 2 spaces
        return text

    @staticmethod
    def remove_hyperlinks(text):
        text = re.sub(r'https?://\S+', '', text)
        return text


    @staticmethod
    def process_jsonl_files(base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.jsonl'):
                    # Determine the human and machine text columns based on filename
                    if 'bloomz' in file:
                        human_text_column = 'machine_text'
                        machine_text_column = 'attack_machine_text'
                    elif 'MT_llama' in file:
                        human_text_column = 'machine_text'
                        machine_text_column = 'attack_machine_text'
                    else:
                        human_text_column = 'machine_text'
                        machine_text_column = 'attack_machine_text'

                    # Path to the JSONL file
                    jsonl_path = os.path.join(root, file)

                    # Create new directory for preprocessed files
                    new_dir = f"{os.path.basename(root)}_preprocessed"
                    new_dir_path = os.path.join(base_path, new_dir)
                    os.makedirs(new_dir_path, exist_ok=True)

                    # Path for new JSONL file
                    new_jsonl_path = os.path.join(new_dir_path, file)
                    # Read and process the JSONL file
                    with open(jsonl_path, 'r', encoding='utf-8') as infile, \
                        open(new_jsonl_path, 'w', encoding='utf-8') as outfile:
                        
                        for line in infile:
                            json_obj = json.loads(line)
                            
                            if human_text_column in json_obj:
                                #print(json_obj[human_text_column])
                                json_obj[human_text_column] = DataPreprocessing.convert_latex_to_text(json_obj[human_text_column])
                                # json_obj[human_text_column] = DataPreprocessing.remove_inline_latex(json_obj[human_text_column])
                                #json_obj[human_text_column] = DataPreprocessing.remove_non_ascii(json_obj[human_text_column])
                                json_obj[human_text_column] = DataPreprocessing.remove_non_printable(json_obj[human_text_column])
                                json_obj[human_text_column] = DataPreprocessing.remove_spaces(json_obj[human_text_column])
                                json_obj[human_text_column] = DataPreprocessing.remove_hyperlinks(json_obj[human_text_column])
                                #print(json_obj[human_text_column])
                            
                            if machine_text_column in json_obj:
                                #print(json_obj[machine_text_column])
                                json_obj[machine_text_column] = DataPreprocessing.convert_latex_to_text(json_obj[machine_text_column])
                                # json_obj[machine_text_column] = DataPreprocessing.remove_inline_latex(json_obj[machine_text_column])
                                #json_obj[machine_text_column] = DataPreprocessing.remove_non_ascii(json_obj[machine_text_column])
                                json_obj[machine_text_column] = DataPreprocessing.remove_non_printable(json_obj[machine_text_column])
                                json_obj[machine_text_column] = DataPreprocessing.remove_spaces(json_obj[machine_text_column])
                                json_obj[machine_text_column] = DataPreprocessing.remove_hyperlinks(json_obj[machine_text_column])
                                #print(json_obj[machine_text_column])
                            # Write the processed JSON object to the new JSONL file
                            outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

# text1 = "In this study, we have attempted to estimate the mass and radius of the unseen M-dwarf companion in the single-lined eclipsing binary, HAT-TR-205-013. Eclipsing binaries provide valuable information about the fundamental properties of stars, such as their masses and radii, which can then be used to test theoretical models. Our main motivation for this research was to improve our understanding of the physical properties of low-mass stars, which are important for a range of astrophysical phenomena, from the formation of planets to the evolution of galaxies.  Using high-precision photometry and spectroscopy, we analyzed the light curve and radial velocity of HAT-TR-205-013 to obtain the binary parameters. From these measurements, we found that the unseen companion has a mass of 0.29\u00b10.04 M\u2609 and a radius of 0.33\u00b10.02 R\u2609, which are consistent with theoretical predictions for an M-dwarf star. Our results also indicate that the primary star has a mass of 0.79\u00b10.03 M\u2609 and a radius of 0.77\u00b10.02 R\u2609.  The accurate determination of the mass and radius of the M-dwarf companion in HAT-TR-205-013 will help to improve our understanding of low-mass stellar evolution and provide a benchmark for testing theoretical models. Our study highlights the importance of continued observations of eclipsing binaries to advance our knowledge of the fundamental properties of stars."
# converted_text1 = DataPreprocessing.convert_unicode_characters(text1)
# print(converted_text1)

# text2 = "We derive masses and radii for both components in the single-lined eclipsing binary HAT-TR-205-013, which consists of a F7V primary and a late M-dwarf secondary. The system's period is short, $P=2.230736 \\pm 0.000010$ days, with an orbit indistinguishable from circular, $e=0.012 \\pm 0.021$. We demonstrate generally that the surface gravity of the secondary star in a single-lined binary undergoing total eclipses can be derived from characteristics of the light curve and spectroscopic orbit. This constrains the secondary to a unique line in the mass-radius diagram with $M/R^2$ = constant. For HAT-TR-205-013, we assume the orbit has been tidally circularized, and that the primary's rotation has been synchronized and aligned with the orbital axis. Our observed line broadening, $V_{\\rm rot} \\sin i_{\\rm rot} = 28.9 \\pm 1.0$ \\kms, gives a primary radius of $R_{\\rm A} = 1.28 \\pm 0.04$ \\rsun. Our light curve analysis leads to the radius of the secondary, $R_{\\rm B} = 0.167 \\pm 0.006$ \\rsun, and the semimajor axis of the orbit, $a = 7.54 \\pm 0.30 \\rsun = 0.0351 \\pm 0.0014$ AU. Our single-lined spectroscopic orbit and the semimajor axis then yield the individual masses, $M_{\\rm B} = 0.124 \\pm 0.010$ \\msun and $M_{\\rm A} = 1.04 \\pm 0.13$ \\msun. Our result for HAT-TR-205-013 B lies above the theoretical mass-radius models from the Lyon group, consistent with results from double-lined eclipsing binaries. The method we describe offers the opportunity to study the very low end of the stellar mass-radius relation."
# converted_text2 = DataPreprocessing.convert_latex_to_text(text2)
# print(converted_text2)

DataPreprocessing.process_jsonl_files('./data/prompt_attack')