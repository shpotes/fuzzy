##
# Fuzzy logic
#
# @file
# @version 0.1

PROJNAME=main

.PHONY: $(PROJNAME).pdf all clean

all: $(PROJNAME).pdf

$(PROJNAME).pdf: $(PROJNAME).tex
	latexmk -bibtex -f -pdf -outdir=build -xelatex -use-make $<

cleanall:
	latexmk -C -outdir=build

clean:
	latexmk -c -outdir=build

# end
