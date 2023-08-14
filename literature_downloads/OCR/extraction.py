# some python file
import textract

# Note issues with columns. Does semantic scholar deal with this?

def example():
    text = textract.process("Pride et al. - 2023 - CORE-GPT Combining Open Access research and large.pdf")
    f = open('example.txt', 'wb')
    f.write(text)
    f.close()

if __name__ == '__main__':
    example()