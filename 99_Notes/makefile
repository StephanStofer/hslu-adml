PARAMS=--pdf-engine=xelatex --metadata-file=$(OPTIONS) --resource-path=.:notes:$(SEARCHPATH)
OPTIONS=document_options.yaml
OUTPUT=Zusammenfassung_ADML_F21
INPUT=notes/*.md
SEARCHPATH=notes/images/
FILEEXTENSION=.pdf

all: compile open

compile: $(TARGET)
	pandoc -s $(PARAMS) $(INPUT) -o $(OUTPUT)$(FILEEXTENSION)

open:
	open $(OUTPUT)$(FILEEXTENSION)
