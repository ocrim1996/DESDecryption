#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "Decrypter.h"

using namespace std;

#define DICTIONARY "dict_1500000.txt"

double execute(Decrypter d, int threads, string password, string salt, int nAverage) {

    cout << "[*] Decrypting using password '" << password << "' and " << threads << " threads, averaging " << nAverage << " times..." << endl;

    try {

        double averageTime;
        double sum = 0;
        for (int i=0; i<nAverage; i++) {
            sum += d.decrypt(threads);
        }
        averageTime = sum/nAverage;
        cout << "[i] Averaged time: " << averageTime << "\n" << endl;
        return averageTime;

    } catch (runtime_error &e) {
        std::cerr << "Caught exception: " << e.what() << endl;
    }
}

int main(int argc, char** argv) {

    string salt = "PC"; // 2 characters salt is needed to use DES in r_crypt() func
    Decrypter d(DICTIONARY, salt);
    int nAverage = 5;

    // FIRST TEST (test all thread nums on three different passwords)

    vector<string> passwords = {"3kingdom", "giacomix", "6Melissa"};
    vector<int> nThreads = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    vector<double> executionTimes;

    for (auto &password: passwords) {

        executionTimes.clear();
        d.setPassword(password);

        for (auto &threads: nThreads) {
            double time = execute(d, threads, password, salt, nAverage);
            executionTimes.push_back(time);
        }
        
        // Print execution times
        cout << "\n\n[*] Times for password '" << password << "': ";
        for (auto const& time: executionTimes)
            std::cout << time << ", ";
        cout << "\n\n" << endl;
    }
    
    // SECOND TEST (10 different passwords using 8 threads)

    /*int threads = 8;
    vector<double> secondExecutionTimes;
    vector<string> passwords = {"sirpizza", "3kingdom", "tyleisha", "marumari", "giacomix", "dbcookie2", "Yessssss", "Mypaypa12", "6Melissa", "1Mazzola"};
    
    for (auto &password: passwords) {
        d.setPassword(password);
        double time = execute(d, threads, password, salt, nAverage);
        secondExecutionTimes.push_back(time);
    }
    
    // Print execution times
    cout << "\n\n[*] Times: ";
    for (auto const& time: secondExecutionTimes)
        std::cout << time << ", ";
    cout << "\n\n" << endl;*/

}
