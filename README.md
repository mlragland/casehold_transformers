# Zero-shot MCQA Evaluation with Legal Pretrained Transformer Models on LexGLUE Case Hold benchmark
Implementing a workflow for evaluating pretrained Transformer models on legal domain specific zero shot multiple choice question and answer tasks using the LexGlue Case Hold benchmark dataset. 

Workflow:
### initialization.ipynb 
this file was used to initialize the test environment for evaluating the legal pretrained Transformer models using LexGLUE CaseHold dataset, based on prior work.  Our endeavor is inspired and builds upon the seminal work presented in the paper "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset" by Lucia Zheng et al. The insights and findings from this paper provide a valuable foundation for our exploration.

### requirements.txt
this file is crucial for encapsulating the dependencies necessary to recreate a Python environment, ensuring consistency across different setups and stremlining the setup process

### casehold.csv
this file has been modified for the purpose of evaluating the dataset against multiple choice.  Describe the changes to the file that are not present in the data available on hugging face.

### data.zip
the data.zip file contains the casehold.csv and overruling.csv files.  These files are located in a google_drive repository and contains the necessary attributes to process the multiple choice questions.  use the code in the notebook to split the data into train,test, dev, and all and load into the data folder


