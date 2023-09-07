# textnetwork_analysis
This is the source code for my thesis on analyzing national AI policies using semantic text network analysis. 
To understand how different governments frame their national AI policies and agenda based on their assumed ideological beliefs, a semantic text network strategy is used. 

The idea behind such strategy is a knowledge map that focuses on the relationships between words found in the national AI policies. These relationships help reveal the structure of the content, extract information and discover underlying meanings and frames in the text. In doing so, cluster of words where words are grouped into a communitiy when they convey similar meanings can also be automatically determined by algorithms, such as the Leiden algorithm used in this thesis. These clusters bring to light the main thems and framings within the policies, which in turn reflect the governments' ideological standings, priorities and perceived challenges of AI. 

This thesis mainly uses the Textnets and NetworkX packages to find and form the map of relationships between words.

# How to make it run
Create a new virtual environment (i.e., pyenv) and install the packages located in the `requirements.txt` file by running `pip install -r requirements.txt`.
Create 3 folders in the same directory of the source code to first place the text documents used `docs`, the prepared docs `processed_docs` and the visualized network graphs `graphs`. 




