

#De local a Amazon

scp -i "~/.ssh/amazon.pem" train.csv ubuntu@ec2-35-178-211-89.eu-west-2.compute.amazonaws.com:~
scp -i "~/.ssh/amazon.pem" test.csv ubuntu@ec2-35-178-211-89.eu-west-2.compute.amazonaws.com:



# De amazon al local

scp -i "~/.ssh/amazon.pem" ubuntu@ec2-35-177-193-234.eu-west-2.compute.amazonaws.com:/home/ubuntu/kaggle_microsoft/Notebooks/lista0.csv /Users/csevilla/Desktop/recibir

scp -i "~/.ssh/amazon.pem" ubuntu@ec2-35-177-193-234.eu-west-2.compute.amazonaws.com:/home/ubuntu/kaggle_microsoft/Notebooks/lista1.csv /Users/csevilla/Desktop/recibir