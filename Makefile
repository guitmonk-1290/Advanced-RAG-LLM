install:
	#install commands
	pip install --upgrade pip &&\
		pip install -r requirements.txt
lint:
	#flake8 or pylint
test:
	#test
deploy:
	#deploy
all: install lint test deploy