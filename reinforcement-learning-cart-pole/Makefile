.PHONY: main
	
title = Actor-Critic Deep Reinforcement Learning: Inverted Pendulum on a Cart
author = Rico A. R. Picone

main: main.pdf # Default document(s) to make

%.txt: %.py
	python $< > $@

%.md: %.py
	publish $< md --title "$(title)" --author "$(author)"

%.tex: %.py
	publish $< tex --title "$(title)" --author "$(author)"
	
%.docx: %.py
	publish $< docx --title "$(title)" --author "$(author)"

%.pdf: %.py
	publish $< pdf --pdflatex True --title "$(title)" --author "$(author)"