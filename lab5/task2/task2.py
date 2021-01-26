"""
Dynamiczny przydział zadań do workerów.

Uruchom obliczanie zbioru Julii wg kodu MPI korzystającego z MPIPoolExecutor
https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#examples
na Zeusie korzytając z liczby rdzeni równej liczbie workerów  MPI (wybranej dowolnie rozsądnie)

1. rozszerz funkcję julia_line(k), tak aby wypisywała ona czas obliczen dla linii, rank procesu oraz numer linii.
2. sprawdz czy czasy różnią się w zależności od numeru linii, policz róznicę między największym, a najmniejszym czasem
3. sprawdz, w jaki sposób linie są przyporządkowywane do procesów - czy można powiedzieć że jest to podział blokowy albo cykliczny?
4. porownaj liczbę linii przyporządkowaną do każdego z procesów workerów
5. Policz sumę czasów dla każdego z procesów workerów. Porównaj te czasy do siebie oraz  do czasu działania całego programu.

6. Obciąż sztucznie obliczenia dla jednej wybranej linii dodając dodatkowe obliczenia w zależności od jej numeru (np. jeśli linia ma nr 1 wykonujemy  funkcję fft() od losowej dużej tablicy.) Wcześniej należy sprawdzić, że czas wykonania "nadmiarowego kodu" jest porónywalny lub większy z sumą czasów dla jednego z workerów obliczoną z p.4 . Sprawdz co się stało wykonując jeszcze raz polecenia 4 i 5

Punktacja:
polecenia 1-5  1pkt
polecenie 6 1 pkt
"""
