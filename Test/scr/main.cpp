#include<iostream>
#include<vector>
#include<random>
#include<algorithm>
#include<map>
#include<unordered_map>
#include<unordered_set>
#include<memory>
#include<functional>
#include<set>
#include<queue>

using namespace std;

//WeightedQuickUnion(加权快速连结)
class WQU {
	vector<int> id;
	vector<int> sz;
	int count = 0;
	int numberOfSite = 0;
public:
	int find(int p);
	void Union(int p, int q);
	int get_count() { return count; }
	bool connected(int p, int q); 
	int Count(int n);
	int newSite();
};
bool WQU::connected(int p, int q) {
	if (p >= numberOfSite || q >= numberOfSite) {
		throw runtime_error("Site does not exist");
	}
	return find(p) == find(q);
}
//压缩路径find，递归，只修改查找时经历节点的连结
int WQU::find(int p) {
	if (p >= numberOfSite) {
		throw runtime_error("Site does not exist");
	}
	if (p == id[p]) { return p; }
	return id[p] = find(id[p]);
}
void WQU::Union(int p, int q) {
	if (p >= numberOfSite || q >= numberOfSite) {
		throw runtime_error("Site does not exist");
	}
	int rp = find(p);
	int rq = find(q);
	if (rp == rq) { return; }
	if (sz[rp] < sz[rq]) {
		id[rp] = rq;
		sz[rq] += sz[rp];
	}
	else {
		id[rq] = rp;
		sz[rp] += sz[rq];
	}
	--count;
}
//测试随机生成数据最终连在一棵树上所需的连结条数
int WQU::Count(int n) {
	while (n > numberOfSite) { newSite(); }
	while (get_count() != 1) {
		int p = rand() % n;
		int q = rand() % n;
		cout << p << ' ' << q << endl;
		if (!connected(p, q)) {
			Union(p, q);
		}
	}
	int cnt = 0;
	for (size_t i = 0; i < id.size(); ++i) {
		if (i != id[i]) {
			++cnt;
		}
	}
	return cnt;
}
//创建一个参与连结的新节点，返回它的值
int WQU::newSite() {
	id.push_back(numberOfSite);
	sz.push_back(1);
	++count;
	return numberOfSite++;
}

//希尔排序
void shellSort(vector<int>& intVe) {
	unsigned stepValue = 1;
	size_t vsize = intVe.size();
	while (3 * stepValue < vsize) { stepValue = stepValue * 3 + 1; }//h = 1,4,13,40...... 
	while (stepValue >= 1) {
		for (unsigned i = stepValue; i < vsize; ++i) {
			for (unsigned j = i; j >= stepValue && intVe[j] < intVe[j - stepValue]; j -= stepValue) {
				swap(intVe[j], intVe[j - stepValue]);
			}
		}
		stepValue /= 3;
	}
}

bool check(vector<int>& intVe) {
	map<int, int>temp;
	for (auto i : intVe) { ++temp[i]; }
	shellSort(intVe);
	for (size_t i = 1; i < intVe.size(); ++i) {
		if (intVe[i - 1] > intVe[i]) { return false; }
	}
	for (auto i : intVe) {
		if (temp.find(i) == temp.end()) { return false; }
		--temp[i];
		if (temp[i] == 0) { temp.erase(i); }
	}
	if (temp.size() != 0) { return false; }
	return true;
}

//原始插入排序
void originInsertSort(vector<int> iv) {
	for (size_t i = 1; i < iv.size(); ++i) {
		for (size_t j = i; i != 0 && iv[j] < iv[j - 1]; --j) {
			swap(iv[j], iv[j - 1]);
		}
	}
}

//哨兵不交换插入排序
void insertSort(vector<int>& intVe) {
	if (intVe.size() == 0) { return; }
	unsigned tempMin = 0;
	for (size_t i = 1; i < intVe.size(); ++i) {
		if (intVe[tempMin] > intVe[i]) {
			tempMin = i;
		}
	}
	swap(intVe[0], intVe[tempMin]);
	for (size_t i = 2; i < intVe.size(); ++i) {
		size_t j = i;
		int temp = intVe[j];
		for (; intVe[j - 1] > intVe[j]; --j) {
			intVe[j] = intVe[j - 1];
		}
		intVe[j] = temp;
	}
}

//归并排序
void merge(vector<int>& vi, size_t lo, size_t mid, size_t hi, vector<int>& aux) {
	size_t begin1 = lo, begin2 = mid;
	for (auto i = lo; i < hi; ++i) {
		if (i < aux.size()) { aux[i] = vi[i]; }
		else { aux.push_back(vi[i]); }
	}
	for (auto i = lo; i < hi; ++i) {
		if (begin1 == mid) { vi[i] = aux[begin2++]; }
		else if (begin2 == hi) { vi[i] = aux[begin1++]; }
		else if (aux[begin2] < aux[begin1]) { vi[i] = aux[begin2++]; }
		else{ vi[i] = aux[begin1++]; }
	}
}
void TopDownMergeSort(vector<int>& vi, size_t lo, size_t hi, vector<int>& aux ) {
	if (lo + 1 == hi) { return; }
	size_t mid = lo + (hi - lo) / 2;
	TopDownMergeSort(vi, lo, mid, aux);
	TopDownMergeSort(vi, mid, hi, aux);
	if (vi[mid] < vi[mid - 1]) { merge(vi, lo, mid, hi, aux); }
}
void DownTopMergeSort(vector<int>& vi, size_t lo, size_t hi, vector<int>& aux) {
	for (size_t i = 1; i < vi.size(); i *= 2) {
		for (size_t j = 0; j < vi.size() - j; j += (2 * i)) {
			merge(vi, j, j + i, min((j + 2 * i), vi.size()), aux);
		}
	}
}
void MergeSort(vector<int>& vi) {
	vector<int> aux;
	if (rand() % 2) { TopDownMergeSort(vi, 0, vi.size(), aux); }
	else { DownTopMergeSort(vi, 0, vi.size(), aux); }
}

//快速排序
void prepare(vector<int>& vi) {
	size_t s = vi.size();
	for (size_t i = 0; i < vi.size(); ++i) { swap(vi[i], vi[rand() % s]); }
	int max = vi[0];
	size_t pmax = 0;
	for (size_t i = 0; i < vi.size(); ++i) {
		if (vi[i] > max) {
			max = vi[i];
			pmax = i;
		}
	}
	swap(vi[pmax], vi[vi.size() - 1]);
}
size_t twoCut(vector<int>& vi, size_t lo, size_t hi) {
	size_t l = lo, h = hi;
	int v = vi[l];
	while (1) {
		while (vi[++l] < v);
		while (v < vi[--h]);
		if (h <= l) { break; }
		swap(vi[l], vi[h]);
	}
	swap(vi[lo], vi[h]);
	return h;
}
void twoQuick(vector<int>& vi, size_t lo, size_t hi) {
	if (lo >= hi-1) { return; }
	size_t j = twoCut(vi, lo, hi);
	twoQuick(vi, lo, j);
	twoQuick(vi, j + 1, hi);
}
void threePrepare(vector<int>& vi) {
	size_t s = vi.size();
	for (size_t i = 0; i < vi.size(); ++i) { swap(vi[i], vi[rand() % s]); }
}
void threeQuick(vector<int>& vi, int lo, int hi) {
	if (lo >= hi) { return; }
	int l = lo, m = lo + 1, h = hi;
	int v = vi[lo];
	while (m <= h) {
		if (vi[m] < v) { swap(vi[l++], vi[m++]); }
		else if (vi[m] > v) { swap(vi[m], vi[h--]); }
		else { ++m; }
	}
	threeQuick(vi, lo, l - 1);
	threeQuick(vi, h + 1, hi);
}

//优先队列
class MaxPQ {
	vector<int> vi = { 0 };
	size_t size = 0;
	void swim(size_t k);
	void sink(size_t k);
public:
	size_t Size() { return size; }
	bool is_empty() { return size == 0; }
	void insert(int n);
	int eraseMax();
};
void MaxPQ::swim(size_t k) {
	while (k > 1 && vi[k / 2] < vi[k]) {
		swap(vi[k / 2], vi[k]);
		k /= 2;
	}
}
void MaxPQ::sink(size_t k) {
	while (k * 2 <= size) {
		size_t k2 = k * 2;
		if (k2 < size && vi[k2] < vi[k2 + 1]) { ++k2; }
		if (vi[k2] <= vi[k]) { break; }
		swap(vi[k], vi[k2]);
		k = k2;
	}
}
void MaxPQ::insert(int n) {
	vi.push_back(n);
	swim(++size);
}
int MaxPQ::eraseMax() {
	int temp = vi[1];
	swap(vi[1], vi[size--]);
	vi.pop_back();
	sink(1);
	return temp;
}

//堆排序
template<typename T>
void sink(vector<T>&vt, size_t l, size_t r) {
	while (2 * l <= r) {
		size_t l2 = 2 * l;
		if (l2 < r && vt[l2] < vt[l2 + 1]) { ++l2; }
		if (vt[l2] <= vt[l]) { break; }
		swap(vt[l2], vt[l]);
		l = l2;
	}
}
template<typename T>
void staticSort(vector<T>&vt) {
	vt.push_back(vt[0]);
	size_t n = vt.size() - 1;
	//生成二叉堆
	for (size_t k = n / 2; k > 0; --k) { sink<T>(vt, k, n); }
	while (n > 1) {
		swap(vt[1], vt[n--]);
		sink<T>(vt, 1, n);
	}
	vt.erase(vt.begin());//erase函数的参数为迭代器或迭代器范围
}


//符号表(双数组实现版)
template<typename K,typename V>
class signTable{
	vector<K> keyV;
	vector<V> valueV;
	size_t n = 0;
public:
	signTable() = default;
	size_t size() { return n; }
	bool is_empty() { return n == 0; }
	K max() { return is_empty() ? NULL : keyV[n - 1]; }
	K min() { return is_empty() ? NULL : keyV[0]; }
	K select(size_t r) { return (r < n) ? keyV[r] : NULL; }
	size_t rank(K key);
	size_t rank(K lo, K hi);
	void insert(K key, V value);
	V get(K key);
	void dele(K key);
	K ceiling(K key);
	K floor(K key);
	vector<K> cut_key(K lo, K hi);
};
template<typename K, typename V>
size_t signTable<K, V>::rank(K key) {
	if (is_empty()) { return 0; }
	int lo = 0, hi = n - 1, mid;
	while (lo <= hi) {
		mid = lo + (hi - lo) / 2;
		if (key < keyV[mid]) { hi = mid - 1; }
		else if (key > keyV[mid]) { lo = mid + 1; }
		else { return mid; }
	}
	return lo;
}
template<typename K, typename V>
size_t signTable<K, V>::rank(K lo,K hi) {
	if (hi <= lo) { return 0; }
	else { return rank(hi) - rank(lo); }
}
template<typename K, typename V>
void signTable<K, V>::insert(K key, V value) {
	size_t temp = rank(key);
	if (temp == n) {
		keyV.push_back(key);
		valueV.push_back(value);
		++n;
	}
	else if (keyV[temp] == key) { valueV[temp] = value; }
	else {
		keyV.push_back(keyV[n - 1]);
		valueV.push_back(valueV[n - 1]);
		for (int i = n - 1; i > temp; --i) {
			keyV[i] = keyV[i - 1];
			valueV[i] = valueV[i - 1];
		}
		keyV[temp] = key;
		valueV[temp] = value;
		++n;
	}
}
template<typename K, typename V>
V signTable<K, V>::get(K key) {
	size_t temp = rank(key);
	if (temp == n || keyV[temp] != key) { return NULL; }
	return valueV[temp];
}
template<typename K, typename V>
void signTable<K, V>::dele(K key) {
	size_t temp = rank(key);
	if (temp == n || keyV[temp] != key) { return; }
	while (temp < n - 1) {
		valueV[temp] = valueV[temp + 1];
		keyV[temp] = keyV[++temp];
	}
	valueV.pop_back();
	keyV.pop_back();
	--n;
}
template<typename K, typename V>
K signTable<K, V>::ceiling(K key) {
	size_t temp = rank(key);
	if (temp == n) { return NULL; }
	return keyV[temp];
}
template<typename K, typename V>
K signTable<K, V>::floor(K key) {
	size_t temp = rank(key);
	if (temp == 0 && keyV[temp] != key) { return NULL; }
	if (temp == n || keyV[temp] != key) { return keyV[temp - 1]; }
	return key; 
}
template<typename K, typename V>
vector<K> signTable<K, V>::cut_key(K lo, K hi) {
	vector<K> temp;
	K temp1 = ceiling(lo);
	if (!temp1) { return temp; }
	size_t temp2 = rank(floor(hi));
	for (size_t i = rank(temp1); i <= temp2; ++i) {
		temp.push_back(keyV[i]);
	}
	return temp;
}

//符号表(二叉搜索树实现版)
class BST {	
	struct Node {
		int key;
		int value;
		shared_ptr<Node> left, right;
		int n;
		Node(int key, int value, int n) :key(key), value(value), n(n) {}
	};
	shared_ptr<Node>root;
	int size(shared_ptr<Node> node) { return node ? node->n : 0; }
	int get(shared_ptr<Node> node, int key) {
		if (node == nullptr) { return 0; }
		if (key < node->key) { return get(node->left, key); }
		else if (key > node->key) { return get(node->right, key); }
		else { return node->value; }
	}
	shared_ptr<Node> put(int key, int value, shared_ptr<Node> node) {
		if (node == nullptr) { node = shared_ptr<Node>(new Node(key, value, 1)); }
		else if (key < node->key) { node->left = put(key, value, node->left); }
		else if (key > node->key) { node->right = put(key, value, node->right); }
		else { node->value = value; }
		node->n = size(node->left) + size(node->right) + 1;
		return node;
	}
	shared_ptr<Node> max(shared_ptr<Node> node) {
		if (node->right == nullptr) { return node; }
		else { return max(node->right); }
	}
	shared_ptr<Node> min(shared_ptr<Node> node) {
		if (node->left == nullptr) { return node; }
		else { return min(node->left); }
	}
	shared_ptr<Node> floor(int key, shared_ptr<Node> node) {
		if (node == nullptr) { return node; };
		if (key < node->key) { return floor(key, node->left); }
		else if (key == node->key) { return node; }
		shared_ptr<Node> tem = floor(key,node->right);
		if (tem) { return tem; }
		return node;
	}
	shared_ptr<Node> ceiling(int key, shared_ptr<Node> node) {
		if (node == nullptr) { return node; };
		if (key > node->key) { return floor(key, node->right); }
		else if (key == node->key) { return node; }
		shared_ptr<Node> tem = floor(key, node->left);
		if (tem) { return tem; }
		return node;
	}
	shared_ptr<Node> select(int ran, shared_ptr<Node> node) {
		if (node == nullptr) { return NULL; }
		int t = size(node->left);
		if (ran < t) { return select(ran, node->left); }
		else if (ran > t) { return select(ran - t - 1, node->left); }
		else { return node; }
	}
	int rank(int key, shared_ptr<Node> node) {
		if (node == nullptr) { return NULL; }
		if (key < node->key) { return rank(key, node->left); }
		else if (key > node->key) { return size(node->left) + 1 + rank(key, node->right); }
		else { return size(node->left); }
	}
	shared_ptr<Node> deleteMax(shared_ptr<Node> node) {
		if (node->right == nullptr) { return node->left; }
		node->right = deleteMax(node->right);
		node->n = size(node->left) + 1 + size(node->right);
		return node;
	}
	shared_ptr<Node> deleteMin(shared_ptr<Node> node) {
		if (node->left == nullptr) { return node->right; }
		node->left = deleteMin(node->left);
		node->n = size(node->left) + 1 + size(node->right);
		return node;
	}
	shared_ptr<Node> deleteKey(shared_ptr<Node> node,int key) {
		if (node == nullptr) { return nullptr; }
		if (key < node->key) { node->left = deleteKey(node->left, key); }
		else if (key > node->key) { node->right = deleteKey(node->right, key); }
		else {
			if (node->left == nullptr) { return node->right; }
			if (node->right == nullptr) { return node->left; }
			shared_ptr<Node>tem = min(node->right);
			node->key = tem->key;
			node->value = tem->value;
			node->right = deleteMin(node->right);
		}
		node->n = size(node->left) + 1 + size(node->right);
		return node;
	}
	vector<int>& keys(shared_ptr<Node>node, vector<int>& vk, int l, int r) {
		if (node == nullptr) { return vk; }
		if (l < node->key) { vk = keys(node->left, vk, l, r); }
		if (l <= node->key && r >= node->key) { vk.push_back(node->key); }
		if (r > node->key) { vk = keys(node->right, vk, l, r); }
		return vk;
	}
public:
	int size() { return size(root); }
	int get(int key) { return get(root, key); }
	void put(int key, int value) { root = put(key, value, root); }
	int max() { return max(root)->key; }
	int min() { return min(root)->key; }
	int floor(int key) { return floor(key, root)->key; }
	int ceiling(int key) { return ceiling(key, root)->key; }
	int select(int ran) { return select(ran, root)->key; }
	int rank(int key) { return rank(key, root); }
	void deleteMax() { root = deleteMax(root); }
	void deleteMin() { root = deleteMin(root); }
	void deleteKey(int key) { root = deleteKey(root, key); }
	vector<int>& keys(vector<int>& vk, int l, int r) { return keys(root, vk, l, r); }
};

//红黑二叉搜索树（左偏）
class RedBlackBST {
	struct RBNode{
		int key, value;
		shared_ptr<RBNode>left, right;
		bool color;
		int n;
		RBNode(int key,int value,int n,bool color):key(key),value(value),n(n),color(color){}
	};
	shared_ptr<RBNode>root;
	bool isRed(shared_ptr<RBNode>node) {
		if (node == nullptr) { return 0; }
		return node->color;
	}
	shared_ptr<RBNode> min(shared_ptr<RBNode> node) {
		if (node->left == nullptr) { return node; }
		else { return min(node->left); }
	}
	int size(shared_ptr<RBNode> node) { return node ? node->n : 0; }
	shared_ptr<RBNode>rotateLeft(shared_ptr<RBNode>node) {
		shared_ptr<RBNode>temp = node->right;
		node->right = temp->left;
		temp->left = node;
		temp->color = node->color;
		node->color = 1;
		temp->n = node->n;
		node->n = size(node->left) + 1 + size(node->right);
		return temp;
	}
	shared_ptr<RBNode>rotateRight(shared_ptr<RBNode>node) {
		shared_ptr<RBNode>temp = node->left;
		node->left = temp->right;
		temp->right = node;
		temp->color = node->color;
		node->color = 1;
		temp->n = node->n;
		node->n = size(node->left) + 1 + size(node->right);
		return temp;
	}
	void flipColors(shared_ptr<RBNode>node) {
		node->color = 1;
		node->left->color = 0;
		node->right->color = 0;
	}
	int get(shared_ptr<RBNode>node, int key) {
		if (node->value > key) { return get(node->left, key); }
		else if (node->value < key) { return get(node->right, key); }
		return node->value;
	}
	shared_ptr<RBNode>put(shared_ptr<RBNode>node, int key, int value) {
		if (node == nullptr) { return node = shared_ptr<RBNode>(new RBNode(key, value, 1, 1)); }
		if (key < node->value) { node->left = put(node->left, key, value); }
		else if (key > node->value) { node->right = put(node->right, key, value); }
		else {
			node->key = key;
			node->value = value;
		}
		if (isRed(node->right) && (!isRed(node->left)) ) { node = rotateLeft(node); }
		if (isRed(node->left) && isRed(node->left->left)) { node = rotateRight(node); }
		if (isRed(node->left) && isRed(node->right)) { flipColors(node); }
		node->n = size(node->left) + 1 + size(node->right);
		return node;
	}
	shared_ptr<RBNode>deleteMin(shared_ptr<RBNode>node) {
		if (node->left == nullptr) { return node->right; }
		if (isRed(node->left)) { node->left = deleteMin(node->left); }
		else {
			if (isRed(node->right)) { node = rotateLeft(node); }
			else { 
				node->color = 0;
				node->left->color = 1;
				if (node->right) { node->right->color = 1; }
			}
			node->left = deleteMin(node->left);
		}
		if (isRed(node->right) && (!isRed(node->left))) { node = rotateLeft(node); }
		if (isRed(node->left) && isRed(node->left->left)) { node = rotateRight(node); }
		if (isRed(node->left) && isRed(node->right)) { flipColors(node); }
		node->n = size(node->left) + 1 + size(node->right);
		return node;
	}
	shared_ptr<RBNode>deleteMax(shared_ptr<RBNode>node) {
		if (node->right == nullptr) { return node->left; }
		if (isRed(node->right)) { node->right = deleteMax(node->right); }
		else {
			if (isRed(node->left)) { node = rotateRight(node); }
			else {
				node->color = 0;
				node->right->color = 1;
				if (node->left) { node->left->color = 1; }
			}
			node->right = deleteMax(node->right);
		}
		if (isRed(node->right) && (!isRed(node->left))) { node = rotateLeft(node); }
		if (isRed(node->left) && isRed(node->left->left)) { node = rotateRight(node); }
		if (isRed(node->left) && isRed(node->right)) { flipColors(node); }
		node->n = size(node->left) + 1 + size(node->right);
		return node;
	}
	shared_ptr<RBNode>deleteKey(shared_ptr<RBNode>node, int key) {
		if (node == nullptr) { return nullptr; }
		if (key < node->key) { 
			if (!isRed(node->left)) {
				if (isRed(node->right)) { node = rotateLeft(node); }
				else {
					node->color = 0;
					node->left->color = 1;
					if (node->right) { node->right->color = 1; }
				}
			}
			node->left = deleteKey(node->left, key); 
		}
		else if (key > node->key) { 
			if (!isRed(node->right)) {
				if (isRed(node->left)) { node = rotateRight(node); }
				else {
					node->color = 0;
					node->right->color = 1;
					if (node->left) { node->left->color = 1; }
				}
			}
			node->right = deleteKey(node->right, key); 
		}
		else {
			if (node->left == nullptr) { return node->right; }
			if (node->right == nullptr) { return node->left; }
			shared_ptr<RBNode>tmp = min(node->right);
			node->key = tmp->key;
			node->value = tmp->value;
			node->right = deleteMin(node->right);
		}
		if (isRed(node->right) && (!isRed(node->left))) { node = rotateLeft(node); }
		if (isRed(node->left) && isRed(node->left->left)) { node = rotateRight(node); }
		if (isRed(node->left) && isRed(node->right)) { flipColors(node); }
		node->n = size(node->left) + 1 + size(node->right);
		return node;
	}
public:
	int size() { return  size(root); }
	int get(int key) { return get(root, key); }
	void put(int key, int value) { 
		root=put(root, key, value);
		root->color = 0;
	}
	int min() { return min(root)->key; }
	void deleteMin() { 
		root = deleteMin(root);
		if (root) { root->color = 0; }
	}
	void deleteMax() { 
		root = deleteMax(root);
		if (root) { root->color = 0; }
	}
	void deleteKey(int key) { 
		root = deleteKey(root, key); 
		if (root) { root->color = 0; }
	}
};

//拉链法散列表（哈希表）
hash<int> h;
class SparateChainingHashST {
	struct Node {
		int key, value;
		Node* next;
		Node(int key,int value):key(key),value(value),next(nullptr){}
	};
	int sz = 0;
	int cpc;
	vector<Node*>sTable;
	int hash(int key) { return h(key) % cpc; }
public:
	SparateChainingHashST(int cpc = 997);
	void put(int key, int value);
	int get(int key);
	void deleteKey(int key);
	~SparateChainingHashST();
};
SparateChainingHashST::SparateChainingHashST(int cpc) : cpc(cpc) {
	for (int i = 0; i < cpc; ++i) { sTable.push_back(nullptr); }
}
void SparateChainingHashST::put(int key, int value) {
	int rank = hash(key);
	if (!sTable[rank]) {
		sTable[rank] = new Node(key, value);
		++sz;
	}
	else{
		Node* tmp = sTable[rank];
		while(1){
			if (tmp->key == key) {
				tmp->value = value;
				return;
			}
			if (tmp->next) { tmp = tmp->next; }
			else { break; }
		}
		tmp->next = new Node(key, value);
		++sz;
	}
}
int SparateChainingHashST::get(int key) {
	Node* tmp = sTable[hash(key)];
	for (; tmp; tmp = tmp->next) {
		if (tmp->key == key) { return tmp->value; }
	}
	return NULL;
}
void SparateChainingHashST::deleteKey(int key) {
	int rank = hash(key);
	Node* tmp;
	if (tmp = sTable[rank]) {	;
		if (tmp->key == key) {
			sTable[rank] = tmp->next;
			delete tmp;
			--sz;
			return;
		}
		while (tmp->next) {
			if (tmp->next->key == key) {
				Node* tmp2 = tmp->next;
				tmp->next = tmp2->next;
				delete tmp2;
				--sz;
				return;
			}
		}
	}
}
SparateChainingHashST::~SparateChainingHashST() {
	for (int i = 0; i < cpc; ++i) { 
		while (sTable[i]) {
			Node* tmp = sTable[i]->next;
			delete sTable[i];
			--sz;
			sTable[i] = tmp;
		}
		if (sz <= 0) { break; }
	}
}

//线性探测法散列表（哈希表）(key不为0)
class LinearProbingHashST {
	int sz = 0, cpc;
	int* Key, * Value;
	int hash(int key) { return h(key) % cpc; }
	void resize(int ncpc);
public:
	LinearProbingHashST(int cpc = 16) :cpc(cpc),Key(new int[cpc]),Value(new int[cpc]){
		for (int i = 0; i < cpc; ++i) {
			Key[i] = 0;
			Value[i] = 0;
		}
	}
	void put(int key, int value);
	int get(int key);
	void deleteKey(int key);
	~LinearProbingHashST() {
		delete[] Key;
		delete[] Value;
	}
};
void LinearProbingHashST::put(int key, int value) {
	if (sz > cpc / 2) { resize(2 * cpc); }
	int temp;
	for (temp = hash(key); Key[temp]; temp = (temp + 1) % cpc) {
		if (Key[temp] == key) {
			Value[temp] = value;
			return;
		}
	}
	Key[temp] = key;
	Value[temp] = value;
	++sz;
}
int LinearProbingHashST::get(int key) {
	for (int temp = hash(key); Key[temp]; temp = (temp + 1) % cpc) {
		if (Key[temp] == key) { return Value[temp]; }
	}
	return NULL;
}
void LinearProbingHashST::deleteKey(int key) {
	int temp = hash(key);
	while (Key[temp]) {
		if (Key[temp] == key) {
			Key[temp] = 0;
			Value[temp] = 0;
			--sz;
			temp = (temp + 1) % cpc;
			while (Key[temp]) {
				int tkey = Key[temp];
				int tvalue = Value[temp];
				Key[temp] = 0;
				Value[temp] = 0;
				--sz;
				put(tkey, tvalue);
				temp = (temp + 1) % cpc;
			}
			break;
		}
		temp = (temp + 1) % cpc;
	}
	if (sz > 0 && sz < cpc / 8) { resize(cpc / 2); }
}
void LinearProbingHashST::resize(int ncpc) {
	int* tKey = new int[ncpc], * tValue = new int[ncpc];
	for (int i = 0; i < ncpc; ++i) {
		tKey[i] = 0;
		tValue[i] = 0;
	}
	for (int i = 0; i < cpc; ++i) {
		if (Key[i]) {
			int trank = hash(Key[i]);
			int temp;
			for (temp = trank; tKey[temp]; temp = (temp + 1) % cpc) {}
			tKey[temp] = Key[i];
			tValue[temp] = Value[i];
		}
	}
	delete[]Key;
	delete[]Value;
	Key = tKey;
	Value = tValue;
	cpc = ncpc;
}

//无向图
class Graph {
	set<int>* adj;
	int v, e = 0;
	bool* marked;//被搜索过的标志
	int* edgeTo;//储存一次搜索中的每个结点的上一个结点的数组
	int* id;//储存所在连通组号
	bool* color;//二分图问题储存颜色
	bool has;//有无环标志
	bool iTC;//是否为二分图标志
	void dfs(int s);
	void dfsForhc(int now, int last);
	void dfsForitc(int v);
	void refresh() { for (int i = 0; i < v; ++i) { marked[i] = 0; } }
public:
	Graph(int v) :v(v), adj(new set<int>[v]), marked(new bool[v]), edgeTo(new int[v]), id(new int[v]), color(new bool[v]) {
		for (int i = 0; i < v; ++i) {
			adj[i] = set<int>();
		}
	}
	int  getv() { return v; }
	int  gete() { return e; }
	void addEdge(int v, int w); 
	set<int>const& Adj(int v) { return adj[v]; }
	bool const* DFS(int s); //深度优先搜索(返回所有点与s的链接情况)
	bool const* BFS(int s);//广度优先搜索(返回所有点与s的链接情况)
	int const* BFP(int s);//单点最短路径问题(广搜法)
	int const* CC();//连通性问题(广搜法)
	bool hasCycle();//检测图中有无环(深搜法)
	bool isTwoColorable();//二分图问题(深搜法)
	~Graph() { delete[]adj; delete[]marked; delete[]edgeTo; delete[]id; delete[]color; }
};
void Graph::addEdge(int v, int w) {
	adj[v].insert(w);
	adj[w].insert(v);
	++e;
}
bool const* Graph::DFS(int s) {
	refresh();
	marked[s] = 1;
	dfs(s);
	return marked;
}
void Graph::dfs(int s) {
	for (auto i : adj[s]) {
		if (!marked[i]) {
			marked[i] = 1;
			dfs(i);
		}
	}
}
bool const* Graph::BFS(int s) {
	refresh();
	queue<int>qi;
	marked[s] = 1;
	qi.push(s);
	while (!qi.empty()) {
		int tem = qi.front();
		qi.pop();
		for (auto i : adj[tem]) {
			if (!marked[i]) {
				marked[i] = 1;
				qi.push(i);
			}
		}
	}
	return marked;
}
int const* Graph::BFP(int s) {
	refresh();
	queue<int>qi;
	marked[s] = 1;
	edgeTo[s] = s;
	qi.push(s);
	while (!qi.empty()) {
		int tem = qi.front();
		qi.pop();
		for (auto i : adj[tem]) {
			if (!marked[i]) {
				edgeTo[i] = tem;
				marked[i] = 1;
				qi.push(i);
			}
		}
	}
	return edgeTo;
}
int const* Graph::CC() {
	refresh();
	int count = 0;
	queue<int>qi;
	for (int i = 0; i < v; ++i) {
		if (!marked[i]) {
			marked[i] = 1;
			id[i] = count;
			qi.push(i);
			while (!qi.empty()) {
				int tem = qi.front();
				qi.pop();
				for (auto j : adj[tem]) {
					if (!marked[j]) {
						marked[j] = 1;
						id[j] = count;
						qi.push(j);
					}
				}
			}
			++count;
		}
	}
	return id;
}
bool Graph::hasCycle() {
	refresh();
	has = 0;
	for (int i = 0; i < v; ++i) {
		if (!marked[i]) {
			marked[i] = 1;
			dfsForhc(i, i);
		}
	}
	return has;
}
void Graph::dfsForhc(int now, int last) {
	for (auto i : adj[now]) {
		if (!marked[i]) {
			marked[i] = 1;
			dfsForhc(i, now);
		}
		else if (i != last) { has = 1; }
	}
}
bool Graph::isTwoColorable() {
	for (int i = 0; i < v; ++i) {
		marked[i] = 0;
		color[i] = 0;
	}
	iTC = 1;
	for (int i = 0; i < v; ++i) {
		if (!marked[i]) {
			marked[i] = 1;
			dfsForitc(i);
		}
	}
	return iTC;
}
void Graph::dfsForitc(int v) {
	for (auto i : adj[v]) {
		if (!marked[i]) {
			marked[i] = 1;
			color[i] = !color[v];
			dfsForitc(i);
		}
		else if (color[i]==color[v]) { iTC = 0; }
	}
}

//有向图
class Digraph {
	set<int>* adj;
	int v, e = 0;
	bool* marked;//被搜索过的标志
	int* edgeTo;//储存一次搜索中的每个结点的上一个结点的数组
	int* id;//储存所在连通组号
	vector<int>cycle;
	bool* onStack;
	queue<int>pre;
	queue<int>post;
	vector<int>reversePost;
	void dfs(int s);
	void dfsFordc(int s);
	void dfo(int s);
	void refresh() { for (int i = 0; i < v; ++i) { marked[i] = 0; } }
public:
	Digraph(int v) :v(v), adj(new set<int>[v]), marked(new bool[v]), edgeTo(new int[v]), id(new int[v]), onStack(new bool[v]) {
		for (int i = 0; i < v; ++i) {
			adj[i] = set<int>();
		}
	}
	int  getv() { return v; }
	int  gete() { return e; }
	void addEdge(int v, int w) { adj[v].insert(w); ++e; }
	set<int>const& Adj(int v) { return adj[v]; }
	bool const* DFS(int s);//深搜
	bool const* BFS(int s);//广搜
	int const* BFP(int s);//单点最短路径
	void DFO();//三种深搜排序（前序，后序，拓扑）
	queue<int>const& getPre() { return pre; }
	queue<int>const& getPost() { return post;}
	vector<int>const& getReversePost() { return reversePost; }
	bool directedCircle();//有向环判断
	int const* SCC();//强连通性
	~Digraph() { delete[]adj; delete[]marked; delete[]edgeTo; delete[]id; delete[]onStack; }
};
bool const* Digraph::DFS(int s) {
	refresh();
	marked[s] = 1;
	dfs(s);
	return marked;
}
void Digraph::dfs(int s) {
	for (auto i : adj[s]) {
		if (!marked[i]) {
			marked[i];
			dfs(i);
		}
	}
}
bool const* Digraph::BFS(int s) {
	refresh();
	queue<int>qi;
	qi.push(s);
	marked[s] = 1;
	while (!qi.empty()) {
		int tem = qi.front();
		qi.pop();
		for (auto i : adj[tem]) {
			if (!marked[i]) {
				marked[i] = 1;
				qi.push(i);
			}
		}
	}
	return marked;
}
int const* Digraph::BFP(int s) {
	refresh();
	queue<int>qi;
	qi.push(s);
	marked[s] = 1;
	edgeTo[s] = s;
	while (!qi.empty()) {
		int tem = qi.front();
		qi.pop();
		for (auto i : adj[tem]) {
			if (!marked[i]) {
				marked[i] = 1;
				edgeTo[i] = tem;
				qi.push(i);
			}
		}
	}
	return edgeTo;
}
bool Digraph::directedCircle() {
	cycle.clear();
	for (int i = 0; i < v; ++i) {
		marked[i] = 0;
		onStack[i] = 0;
	}
	for (int i = 0; i < v; ++i) {
		if (!marked[i]) { 
			marked[i] = 1;
			dfsFordc(i);
		}
	}
	return !cycle.empty();
}
void Digraph::dfsFordc(int s) {
	onStack[s] = 1;
	for (auto i : adj[s]) {
		if (!cycle.empty()) { return; }
		if (!marked[i]) {
			marked[i] = 1;
			edgeTo[i] = s;
			dfsFordc(i);
		}
		else if(onStack[i]){
			for (int tem = i; tem != s; tem = edgeTo[tem]) { cycle.push_back(tem); }
			cycle.push_back(s);
			cycle.push_back(i);
		}
	}
	onStack[s] = 0;
}
void Digraph::DFO() {
	refresh();
	for (int i = 0; i < v; ++i) {
		if (!marked[i]) {
			marked[i] = 1;
			dfo(i);
		}
	}
}
void Digraph::dfo(int s) {
	pre.push(s);
	for (auto i : adj[s]) {
		if (!marked[i]) {
			marked[i] = 1;
			dfo(i);
		}
	}
	post.push(s);
	reversePost.push_back(s);
}
int const* Digraph::SCC() {
	if (directedCircle()) { return nullptr; }
	DFO();
	refresh();
	queue<int>qi;
	int count = 0;
	for (int j = reversePost[reversePost.size() - 1]; j >= 0; --j) {
		int tem = reversePost[j];
		if (!marked[tem]) {
			marked[tem] = 1;
			id[tem] = count;
			qi.push(tem);
			while (!qi.empty()) {
				int temp = qi.front();
				qi.pop();
				for (auto i : adj[temp]) {
					if (!marked[i]) {
						marked[i] = 1;
						id[i] = count;
						qi.push(i);
					}
				}
			}
			++count;
		}
	}
	return id;
}

//加权有向图
class EdgeWeightedDiraph {
	struct DirectedEdge {
		int v;
		int w;
		double weight;
		DirectedEdge(int v,int w,int weight):v(v),w(w),weight(weight){}
	};
	int V, E = 0;
	vector<DirectedEdge>* adj;
	vector<int>reversePost;
	vector<int>cycle;
	bool* marked;
	bool* onStack;
	int* edgeTo;
	double* distTo;
	bool* onQ;
	int cost;//BF放松次数
	bool hasCycle;
	deque<int>qForBF;
	void refresh() { for (int i = 0; i < V; ++i) { marked[i] = 0; } }
	void relex(int v);
	void relexForBF(int v);
	void findNegativeCycle();
	void dfoFortp(int s);//为实现拓扑序的深搜
	void directedCircle();//检查有向环
	void dfsFordc(int s);//为查找有向环的深搜
public:
	EdgeWeightedDiraph(int v) :V(v), adj(new vector<DirectedEdge>[v]), marked(new bool[v]), onStack(new bool[v]), edgeTo(new int[v]), distTo(new double[v]), onQ(new bool[v]) {
		for (int i = 0; i < v; ++i) {
			adj[i] = vector<DirectedEdge>();
		}
	}
	int getV() { return V; }
	int getE() { return E; }
	void addEdge(int v, int w, int weight) {
		adj[v].push_back(DirectedEdge(v, w, weight));
		++E;
	}
	vector<DirectedEdge> const* const Adj(int v) { return &adj[v]; }
	int const* const get_edgeTo() { return edgeTo; }
	double const* const get_distTo() { return distTo; }
	void AcyclicSP(int s);//获取无环有向图的方法(按拓扑序放松所有结点)
	void BellmanFordSP(int s);//获取最短路径的一般方法(Bellman-Ford算法FIFO改进)
	~EdgeWeightedDiraph() { delete[]adj; delete[]marked; delete[]onStack; delete[]edgeTo; delete[]distTo; delete[]onQ; }
};
void EdgeWeightedDiraph::relex(int v) {
	for (int I = 0; I < adj[v].size(); ++I) {
		DirectedEdge const& i = adj[v][I];
		int w = i.w;
		if (distTo[w] > distTo[v] + i.weight) {
			distTo[w] = distTo[v] + i.weight;
			edgeTo[w] = v;
		}
	}
}
void EdgeWeightedDiraph::relexForBF(int v) {
	for (int I = 0; I < adj[v].size(); ++I) {
		DirectedEdge const& i = adj[v][I];
		int w = i.w;
		if (distTo[w] > distTo[v] + i.weight) {
			distTo[w] = distTo[v] + i.weight;
			edgeTo[w] = v;
			if (!onQ[w]) {
				qForBF.push_back(w);
				onQ[w] = 1;
			}
		}
	}
	if (cost++ == V) {
		findNegativeCycle();
	}
}
void EdgeWeightedDiraph::dfoFortp(int s) {
	for (int I = 0; I < adj[s].size(); ++I) {
		DirectedEdge const& i = adj[s][I];
		int w = i.w;
		if (!marked[w]) {
			marked[w] = 1;
			dfoFortp(w);
		}
	}
	reversePost.push_back(s);
}
void EdgeWeightedDiraph::AcyclicSP(int s) {
	for (int i = 0; i < V; ++i) {
		distTo[i] = DBL_MAX;
		marked[i] = 0;
	}
	distTo[s] = 0.0;
	marked[s] = s;
	for (int i = 0; i < V; ++i) {
		if (!marked[i]) {
			marked[i] = 1;
			dfoFortp(i);
		}
	}
	for (int i = reversePost.size() - 1; i >= 0; --i) { relex(i); }
}
void EdgeWeightedDiraph::directedCircle() {
	cycle.clear();
	for (int i = 0; i < V; ++i) {
		marked[i] = 0;
		onStack[i] = 0;
	}
	for (int i = 0; i < V; ++i) {
		if (!marked[i]) {
			marked[i] = 1;
			dfsFordc(i);
		}
	}
}
void EdgeWeightedDiraph::dfsFordc(int s) {
	onStack[s] = 1;
	for (int I = 0; I < adj[s].size(); ++I) {
		DirectedEdge const& i = adj[s][I];
		if (!cycle.empty()) { return; }
		if (!marked[i.w]) {
			marked[i.w] = 1;
			edgeTo[i.w] = s;
			dfsFordc(i.w);
		}
		else if (onStack[i.w]) {
			for (int tem = i.w; tem != s; tem = edgeTo[tem]) { cycle.push_back(tem); }
			cycle.push_back(s);
			cycle.push_back(i.w);
		}
	}
	onStack[s] = 0;
}
void EdgeWeightedDiraph::BellmanFordSP(int s) {
	qForBF.clear();
	cycle.clear();
	for (int i = 0; i < V; ++i) {
		onQ[i] = 0;
		distTo[i] = DBL_MAX;
		edgeTo[i] = -1;
	}
	distTo[s] = 0.0;
	qForBF.push_back(s);
	onQ[s] = 1;
	hasCycle = 0;
	while (!qForBF.empty() && !hasCycle) {
		int v = qForBF.front();
		qForBF.pop_front();
		onQ[v] = 0;
		relexForBF(v);
	}
}
void EdgeWeightedDiraph::findNegativeCycle() {
	EdgeWeightedDiraph tem(V);
	for (int i = 0; i < V; ++i) {
		if (edgeTo[i] != -1) {
			tem.addEdge(edgeTo[i], i, 1);//边长不重要
		}
	}
	tem.directedCircle();
	hasCycle = !tem.cycle.empty();
}

//三向字符串快排
void Quick3sort(string* s, int lo, int hi, int d) {
	if (lo >= hi) { return; }
	int l = lo, h = hi, i = lo + 1;
	int v = s[lo][d];
	while (i <= h) {
		int tem = s[i][d];
		if (tem < v) { swap(s[i++], s[l++]); }
		else if (tem > v) { swap(s[i], s[h--]); }
		else { ++i; }
	}
	Quick3sort(s, lo, l - 1, d);
	if (v) { Quick3sort(s, l, h, d + 1); }
	Quick3sort(s, h + 1, hi, d);
}
int main() {
	vector<int>vi{ 5,3,7,7,7,3,3,3,7,2,4,60,8,1,2,81,3,48,33 };
	string s[8]{ "fdsa","dsfsd","dfs","few","rwfds","wdafs","fdsf","rfds" };
	Quick3sort(s, 0, 7, 0);
	for (int i = 0; i < 8; ++i) {
		cout << s[i] << "\n";
	}
	return 0;
}