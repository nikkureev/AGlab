def fastaParser(infile):
    seqs = []
    headers = []
    with open(infile, 'r') as f:
        sequence = ""
        header = None
        for line in f:
            if line.startswith('>'):
                headers.append(line[1:-1])
                if header:
                    seqs.append(sequence)
                sequence = ""
                header = line[1:]
            else:
                sequence += line.rstrip()
        seqs.append(sequence)
    return headers, seqs


def FASTA(filename):
    try:
        f = open(filename,"r")
    except IOError:
        print ('The file, {}, does not exist'.format(filename))
        return

    order = []
    sequences = {}

    for line in f:
        if line.startswith('>'):
            name = line[1:].rstrip('\n')
            name = name.replace('_',' ')
            order.append(name)
            sequences[name] = ''

        else:
            sequences[name] += line.rstrip('\n').rstrip('*')

    print ('{} sequences found'.format(len(order)))
    return order, sequences


FASTA('C:/AGlab/one.txt')
