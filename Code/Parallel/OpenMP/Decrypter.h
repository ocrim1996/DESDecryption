#ifndef OPENMPDESDECRYPTION_DECRYPTER_H
#define OPENMPDESDECRYPTION_DECRYPTER_H


#include <string>
#include <vector>

using namespace std;

class Decrypter {
public:
    Decrypter(const string& dictionaryPath, const string& salt);
    double decrypt(int threads);

private:
    string encrypted;
public:
    void setPassword(const string &password);

private:
    string salt;
    vector<string> fullDictionary;
};


#endif //OPENMPDESDECRYPTION_DECRYPTER_H