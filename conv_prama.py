i = 205
o = 102
for k in range(2, 7):
    for p in range(0, 4):
        for s in range(1, 4):
            # if (o + 2 * p - k) % s == 0:
            #     if o == s * (i - 1) - 2 * p + k:
            #         print('k:{}, p:{}, s:{}'.format(k, p, s))
            # else:
            #     if o == s * (i - 1) - 2 * p + k + (o + 2 * p - k) % s:
            #         print('k:{}, p:{}, s:{}'.format(k, p, s))

            if o == (i - k + 2 * p) // s + 1:
                print('k:{}, p:{}, s:{}'.format(k, p, s))