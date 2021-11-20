from data_processing import etlstream
from data_processing.etlstream import StreamFactory


def main():
    st11_ids = set()
    sb_ids = set()
    mc7_ids = set()
    mc2_ids = set()

    sets = [st11_ids, sb_ids, mc7_ids, mc2_ids]
    origins = [etlstream.Origin.ST11, etlstream.Origin.SB, etlstream.Origin.MC7, etlstream.Origin.MC2]

    for i in range(len(origins)):
        stream = StreamFactory.create(origins[i], use_cache=True)
        for p in stream.stream_from_source():
            sets[i].add(p.patient_id)

    sb_ids2 = set(['SC-HF-I-1',
                   'SC-HF-I-2',
                   'SC-HF-I-4',
                   'SC-HF-I-5',
                   'SC-HF-I-6',
                   'SC-HF-I-7',
                   'SC-HF-I-8',
                   'SC-HF-I-9',
                   'SC-HF-I-10',
                   'SC-HF-I-11',
                   'SC-HF-I-12',
                   'SC-HF-I-40',
                   'SC-HF-NI-3',
                   'SC-HF-NI-4',
                   'SC-HF-NI-7',
                   'SC-HF-NI-11',
                   'SC-HF-NI-12',
                   'SC-HF-NI-13',
                   'SC-HF-NI-14',
                   'SC-HF-NI-15',
                   'SC-HF-NI-31',
                   'SC-HF-NI-33',
                   'SC-HF-NI-34',
                   'SC-HF-NI-36',
                   'SC-HYP-1',
                   'SC-HYP-3',
                   'SC-HYP-6',
                   'SC-HYP-7',
                   'SC-HYP-8',
                   'SC-HYP-9',
                   'SC-HYP-10',
                   'SC-HYP-11',
                   'SC-HYP-12',
                   'SC-HYP-37',
                   'SC-HYP-38',
                   'SC-HYP-40',
                   'SC-N-2',
                   'SC-N-3',
                   'SC-N-5',
                   'SC-N-6',
                   'SC-N-7',
                   'SC-N-9',
                   'SC-N-10',
                   'SC-N-11',
                   'SC-N-40'])

    print(st11_ids.intersection(sb_ids))
    print(st11_ids.intersection(sb_ids2))
    print(st11_ids.intersection(mc2_ids))
    print(st11_ids.intersection(mc7_ids))
    print(sb_ids.intersection(mc2_ids))
    print(sb_ids.intersection(mc7_ids))
    print(sb_ids2.intersection(mc2_ids))
    print(sb_ids2.intersection(mc7_ids))
    print(mc2_ids.intersection(mc7_ids))


if __name__ == "__main__":
    main()
