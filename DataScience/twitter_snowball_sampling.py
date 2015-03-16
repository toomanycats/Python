
"""This is a snowball sampler. """
import networkx as net
import glob
import time
import os
import sys
import json
import tweepy
import ipdb

CONSUMER_KEY = "XXXXXXXXXX"
CONSUMER_SECRET = "XXXXXXXXXXX"

ACCESS_TOKEN = "XXXXXXXXXX"
TOKEN_SECRET = "XXXXXXXXXX"

ROOT = '/Users/username/git/Python2.7/resources/'
CENTERS = os.path.join(ROOT, 'centers')
FOLLOWING_DIR = os.path.join(ROOT, 'following')
EDGE_FILE = os.path.join(ROOT, 'twitter_edges.csv')

MAX_FOLLOWERS = 20
MAX_DEPTH = 5

enc = lambda x: x.encode('ascii', errors='ignore')

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, TOKEN_SECRET)
api = tweepy.API(auth)


def main():
    screen_name = "WholeFoods"

    if not os.path.exists(FOLLOWING_DIR):
        os.makedirs(FOLLOWING_DIR)
        os.makedirs(CENTERS)

    #ipdb.set_trace()
    get_user_dict(screen_name, depth=0)

def _get_userfname(center, type_='center'):
    if type_ == "center":
        userfname = os.path.join(CENTERS, enc(center) + '.json')

    elif type_ == "followers":
        userfname = os.path.join(FOLLOWING_DIR, enc(center) + '.json')

    return userfname

def _get_followers_from_file(fname, type_='id'):
    screen_name = os.path.split(fname)[-1]
    print 'Retrieving friends for user: %s' %screen_name
    f = open(fname)
    if type_ == 'id':
        info = [line.strip().split('\t')[1] for line in f]
    elif type_ == 'all':
       info = [line.strip().split('\t') for line in f]
    else:
        raise Exception("Not a valid option")

    f.close()

    # consistent at return
    if len(info) == 0:
        return None

    return info

def _get_list_of_friends(user_id):
    try:
        c = tweepy.Cursor(api.followers, id=user_id).items()

    except tweepy.TweepError, error:
        print error

        wait_for_limit(error)
        _get_list_of_friends(user_id)

    except Exception, err:
        print err
        return None

    return c

def _get_user(center):

    try:
        user = api.get_user(center)
        return user

    except tweepy.TweepError, error:
        print error

        wait_for_limit(error)
        user = _get_user(center)

    except Exception, err:
        print err

        return None

def wait_for_limit(error):
    #ipdb.set_trace()
    try:
        if error[0][0]['message'] == 'Rate limit exceeded':
            print 'Rate limited. Sleeping for 15 minutes.'
            time.sleep(15 * 60 + 15)

    except:
        print str(error)
        return

def get_user_dict(center, depth):
    print 'Retrieving user details for twitter id %s' % enc(center)

    fname = _get_userfname(center, 'center')
    if os.path.exists(fname):
        user = json.loads(file(fname).read())

    else:
        user_info = _get_user(center)

        try:# follwers ids can fail for permission error
            user = {'name': enc(user_info.name),
                    'screen_name': enc(user_info.screen_name),
                    'id': user_info.id,
                    'friends_count': user_info.friends_count,
                    'followers_count': user_info.followers_count,
                    'followers_ids': user_info.followers_ids()
                   }
        except tweepy.TweepError, error:
            wait_for_limit(error)
            return

        with open(fname, 'w') as outf:
            outf.write(json.dumps(user) + '\t')

    fname = _get_userfname(user['screen_name'], 'followers')
    if os.path.exists(fname):
        return
        #followers = _get_followers_from_file(fname)

    else:
        lookup_followers(user['id'], user['screen_name'])
        followers = _get_followers_from_file(fname)

    if followers is not None and depth < MAX_DEPTH * len(followers):
        for follower in followers:
            get_user_dict(follower, depth)
            depth += 1

def lookup_followers(user_id, screen_name):
    cursor = _get_list_of_friends(user_id)

    if cursor is None:
        return None

    fname = _get_userfname(screen_name, 'followers')

    outf = open(fname, 'w')

    cnt = 0
    while True:
        try:
            follower = cursor.next()
            string = '%s\t%s\t%s\n' %(follower.id,
                                      enc(follower.screen_name),
                                      enc(follower.name)
                                      )
            outf.write('%s' %string)
            cnt += 1
            print cnt

            if cnt == MAX_FOLLOWERS:
                break

        except tweepy.TweepError, error:
            wait_for_limit(error)

        # add exception for cursor.next()
        except StopIteration:
            break

    outf.close()


class Graph(object):
    def __init__(self):
        self.users = {'followers': 0}
        self.edges = []

    def load_centers(self):
        for file_ in glob.glob(os.path.join(CENTERS, '*.json')):
            f = open(file_, 'r')
            data = json.load(f)
            f.close()
            screen_name = data['screen_name']
            self.users[screen_name] = {'num_followers': data['followers_count']}

    def build_edges(self, screen_name):
        fname = _get_userfname(screen_name, type_='followers')
        if not os.path.exists(fname):
            return

        followers_info = _get_followers_from_file(fname, type_='all')

        if followers_info is None or len(followers_info) < 2:
            return

        data = self._build_new_data(screen_name, followers_info)
        if data is not None:
            self.edges += (data)

        for item in followers_info:
            self.build_edges(item[1])

    def _build_new_data(self, screen_name, followers):
        new_data = []
        for follower_data in followers:
            screen_name_follower = follower_data[1]

            try: # somtimes the prog was quit early and data is incomplete
                weight = self.users[screen_name_follower]['num_followers']

            except KeyError:
                weight = 1

            new_data.append([screen_name, screen_name_follower, weight])


        return new_data

    def write_edge_dict_to_file(self):
        f = open(EDGE_FILE, 'w')

        for edge in self.edges:
            f.write('%s\t%s\t%d\n' % (edge[0], edge[1], edge[2]))

        f.close()

    def main(self):
        #ipdb.set_trace()
        self.load_centers()
        SEED = 'WholeFoods'
        self.build_edges(SEED)
        self.write_edge_dict_to_file()

class PlotGraph(object):
    def __init__(self):
        self.di_graph = net.DiGraph()
        self.weights = {}

    def load_network(self):
        network = _get_followers_from_file(EDGE_FILE, 'all')

        for screen_name, followed_by, weight in network:
            self.di_graph.add_edge(screen_name, followed_by, int(weight))
            self.weights[screen_name] = int(weight)

    def main(self):
        SEED = "WholeFoods"
        graph = net.DiGraph(net.ego_graph(self.di_graph, SEED, radius=4))

        net.draw_networkx_nodes()
        net.draw_networkx_edges()



if __name__ == "__main__":
    main()
