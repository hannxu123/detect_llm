from dataset2 import Corpus_all

def test(name):
    real, fake = Corpus_all(name)
    print("........................")
    print(name + ":")
    print( str(len(real) ))
    print( str(len(fake)))
    print("sample real:" + str(real[987]))
    print("sample fake:" + str(fake[985]))
    print("........................")

test("World")
test("Sports")
test("Business")

test("IMDb")
test("Amazon")

test("Ivypanda")

test("Eli5")
test("AskSci.")
test("AskHist.")
