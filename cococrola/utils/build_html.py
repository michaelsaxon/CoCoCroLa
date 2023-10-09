

BASE = '''
<html>
<head>
<style>
img   {width: 100px; height: 100px;}
table    {margin: auto;}
</style>
</head>
<body>

  <header style="border-top:2px solid black; border-bottom: 1px dotted #676767; position:fixed; height: 60px; overflow:hidden; top:0; right: 0; width:100%; background-color:white;">
    <!--div class="menucontainer" -->
    <div>
    <h2 style="margin-left: 20pt; margin-top:6pt;float:left;color:black;">
            <a style="color:darkblue; text-weight: 800; font-style: italic;" href="../index.html">CoCo-CroLa Demo Home</a> /
            <a style="color:cD0905; text-weight: 400;" href="index.html">###NAME###</a></h2>
  </div>
  </header>

  <div style="height:65px;">&nbsp;</div> 

<table>
'''

TAIL = '''
</table>
</body>
</html>
'''



def build_squares(input_csv, fname_base=".", base_word_point=0):
    input_csv = open(input_csv, "r").readlines()
    index = input_csv[0].strip().split(",")

    middle = ""

    for line_no, line in enumerate(list(input_csv)[1:]):
        this_line = ""
        line = line.strip().split(",")
        for idx in range(len(index)):
            lang = index[idx]

            # say the concept, language pair
            this_line += f'<h3>{lang},{line[0]}</h3><br><div align="center">'
            for i in range(9):
                if i % 3 == 0:
                    this_line += "<br>"
                fname = f"{fname_base}/{line_no}-{index[idx]}-{line[base_word_point]}-{i}.jpg"
                this_line += f'<img src="{fname}">'
            this_line += "</div>"
        middle += this_line

    return BASE+middle+TAIL



def build_columns(prompts_base, fname_base=".", base_word_point=0):
    prompts_base = open(prompts_base, "r").readlines()
    index = prompts_base[0].strip().split(",")

    middle = ""

    for line_no, line in enumerate(list(prompts_base)[1:]):
        this_line = '<tr style="background-color: black; color: white;"><td style="background-color: white;"></td>'
        line = line.strip().split(",")
        for idx in range(len(index)):
            lang = index[idx]
            word = line[idx]
            if lang == default_index_lang:
                url = "index.html"
            else: 
                url = f"sort_{lang}.html"
            if lang == page_sort_language:
                this_line += f'<td style="text-align: center; background-color: darkred;">{lang}<br>{word}</td>'
            else:
                this_line += f'<td style="text-align: center;"><a style="color: white;" href="{url}">{lang}<br>{word}</td>'
        this_line += "</tr>\n<tr>\n"
        this_line += f'<td style="border-right: 5pt solid rgb(150,0,0); color: rgb(150,0,0); font-size: 20pt; background-color: rgb(255,240,240);"><div style="transform:rotate(-90deg); margin: 0px; padding: 0px;"><b>{line[0]}</b></div></td>'
        # the images
        for idx in range(len(index)):
            lang = index[idx]
            if lang == default_index_lang:
                url = "index.html"
            else: 
                url = f"sort_{lang}.html"
            this_line += "<td>"
            # build a prompt based on the above templates from the 
            word = index[idx]
            for i in range(10):
                fname = f"{line_no}-{index[idx]}-{line[0]}-{i}.jpg"
                if lang != page_sort_language:
                    this_line += f'<a href="{url}">'
                this_line += f'<img src="{fname}">'
                if lang != page_sort_language:
                    this_line += "</a>"
                this_line += '<br>\n'
            this_line += "</td>\n"
        this_line += "</tr>\n"
        middle += this_line

    return BASE+middle+TAIL
