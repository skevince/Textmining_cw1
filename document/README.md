# Textmining coursework 1
# please NOTE that Pytorch version must be 1.2.0   (below 1.3.0 may be OK) !!!!!!

### You can train model by by issuing the following command:
#### % python ./src/question_classifier.py --train --glove --bilstm --finetune --config data/model.config
#####'--glove' or '--random'
#####'--bilstm' or '--bow' or '--bowandlstm'  
###### Note:--bowandlstm can only combine '--glove' (because of the dimention problem)
#####'--finetune' or '--freeze'
### You can test model by by issuing the following command:
#### % python ./src/question_classifier.py --test --glove--config data/model.config --confu_matrix
#####'--glove' or '--random'  
###### Note:should be same as train 
##### '--confu_matrix' is optional
### You can test ensembel model by by issuing the following command:
#### % python ./src/question_classifier.py --ensemble --config data/model.config --confu_matrix
##### '--confu_matrix' is optional