
"""This is a snowball sampler. """
import matplotlib.pyplot as plt
import networkx as nx
import glob
import time
import os
import json
import tweepy
import ipdb
import numpy as np

CONSUMER_KEY = "XXXX"
CONSUMER_SECRET = "XXXX"

ACCESS_TOKEN = "XXXX"
TOKEN_SECRET = "XXXX"

ROOT = '/Users/username/git/Python2.7/resources/'
CENTERS = os.path.join(ROOT, 'centers')
FOLLOWING_DIR = os.path.join(ROOT, 'following')
EDGE_FILE = os.path.join(ROOT, 'twitter_edges.csv')

SEED = 'WholeFoods'
MAX_FOLLOWERS = 50
MAX_DEPTH = 5

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, TOKEN_SECRET)
api = tweepy.API(auth)

enc = lambda x: x.encode('ascii', errors='ignore')

def _get_userfname(center, type_='center'):
    if type_ == "center":
        userfname = os.path.join(CENTERS, enc(center) + '.json')

    elif type_ == "followers":
        userfname = os.path.join(FOLLOWING_DIR, enc(center) + '.json')

    return userfname

def _get_followers_from_file(fname, type_='id'):
    screen_name = os.path.split(fname)[-1]
    f = open(fname)

    if type_ == 'id':
        try:
            info = [line.strip().split('\t')[1] for line in f]
        except IndexError:
            print "No screen name in file:%s" %fname
            info = []

    elif type_ == 'all':
        info = [line.strip().split('\t') for line in f]

    else:
        raise Exception("Not a valid option")

    f.close()

    # consistent at return
    if len(info) == 0:
        return None

    return info


class Sample(object):
    def __init__(self):
        pass

    def main(self):
        screen_name = "WholeFoods"

        if not os.path.exists(FOLLOWING_DIR):
            os.makedirs(FOLLOWING_DIR)
            os.makedirs(CENTERS)

        #ipdb.set_trace()
        self.get_user_dict(screen_name, depth=0)

    def _get_list_of_friends(self, user_id):
        try:
            c = tweepy.Cursor(api.followers, id=user_id).items()

        except tweepy.TweepError, error:
            print error

            wait_for_limit(error)
            self._get_list_of_friends(user_id)

        except Exception, err:
            print err
            return None

        return c

    def _get_user(self, center):

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

    def wait_for_limit(self, error):
        #ipdb.set_trace()
        try:
            if error[0][0]['message'] == 'Rate limit exceeded':
                print 'Rate limited. Sleeping for 15 minutes.'
                time.sleep(15 * 60 + 15)

        except:
            print str(error)
            return

    def get_user_dict(self, center, depth):
        print '\nRetrieving user details for twitter id %s\n' % enc(center)

        fname = _get_userfname(center, 'center')
        if os.path.exists(fname):
            user = json.loads(file(fname).read())

        else:
            user_info = self._get_user(center)

            try:# follwers ids can fail for permission error
                user = {'name': enc(user_info.name),
                        'screen_name': enc(user_info.screen_name),
                        'id': user_info.id,
                        'friends_count': user_info.friends_count,
                        'followers_count': user_info.followers_count,
                        'followers_ids': user_info.followers_ids()
                    }
            except tweepy.TweepError, error:
                self.wait_for_limit(error)
                depth -= 1
                return depth

            with open(fname, 'w') as outf:
                outf.write(json.dumps(user) + '\t')

        fname = _get_userfname(user['screen_name'], 'followers')
        if os.path.exists(fname):
            return
            #followers = _get_followers_from_file(fname)

        else:
            print 'Retrieving friends for user: %s' %center
            self.lookup_followers(user['id'], user['screen_name'])
            followers = _get_followers_from_file(fname)

        if followers is not None and depth < MAX_DEPTH:# * len(followers):
            for follower in followers:
                depth += 1
                self.get_user_dict(follower, depth)
                print "depth: %i" %depth
        #else:
        #    for follower in followers:
        #        print "reached depth:%i" %depth
        #        print 'Retrieving friends for user: %s' %center
        #        self.lookup_followers(user['id'], user['screen_name'])

        return

    def lookup_followers(self, user_id, screen_name):
        cursor = self._get_list_of_friends(user_id)

        if cursor is None:
            return None

        fname = _get_userfname(screen_name, 'followers')

        outf = open(fname, 'w')

        cnt = 0
        while True:
            try:
                follower = cursor.next()
                cnt += 1
                string = '%s\t%s\t%s\n' %(follower.id,
                                        enc(follower.screen_name),
                                        enc(follower.name)
                                        )
                outf.write('%s' %string)
                print "follower:%s" %follower.screen_name

                if cnt == MAX_FOLLOWERS:
                    break

            except tweepy.TweepError, error:
                self.wait_for_limit(error)

            # add exception for cursor.next()
            except StopIteration:
                break

        outf.close()


class Edges(object):
    def __init__(self):
        self.users = {'followers': 0}
        self.edges = []
        self.black_list = []

    def load_users(self):
        for file_ in glob.glob(os.path.join(CENTERS, '*.json')):
            f = open(file_, 'r')
            data = json.load(f)
            f.close()
            screen_name = data['screen_name']
            self.users[screen_name] = {'num_followers': data['followers_count']}

    def build_edges(self, screen_name):
        if screen_name in self.black_list:
            return
        else:
            self.black_list.append(screen_name)

        fname = _get_userfname(screen_name, type_='followers')
        if not os.path.exists(fname):
            return

        followers_info = _get_followers_from_file(fname, type_='all')

        if followers_info is None or len(followers_info) < 2:
            return

        data = self._build_row(screen_name, followers_info)
        if data is not None:
            self.edges += (data)
        else:
            return

        for id, screen_name, user_name in followers_info:
            self.build_edges(screen_name)

    def _build_row(self, screen_name, followers):
        new_data = []
        for follower_data in followers:
            screen_name_follower = follower_data[1]

            try: # somtimes the prog was quit early and data is incomplete
                weight = self.users[screen_name_follower]['num_followers']

            except KeyError:
                weight = 1

            new_data.append([screen_name, screen_name_follower, weight])


        return new_data

    def write_edges_to_file(self):
        f = open(EDGE_FILE, 'w')

        for edge in self.edges:
            f.write('%s\t%s\t%d\n' % (edge[0], edge[1], edge[2]))

        f.close()

    def main(self):
        ipdb.set_trace()
        self.load_users()
        self.build_edges(SEED)
        self.write_edges_to_file()


class Network(object):
    def __init__(self):
        self.di_graph = nx.DiGraph()
        self.weights = {}

    def load_network(self):
        network = _get_followers_from_file(EDGE_FILE, 'all')

        for screen_name, followed_by, weight in network:
            weight = int(weight)
            self.di_graph.add_edge(screen_name, followed_by, weight=weight)
            self.weights[screen_name] = weight

    def scale_positions(self, pos):
        # tinest bit faster than a
        # list comprehension with dict()
        # and probably more clear
        for k, v in pos.iteritems():
            pos[k] = 5 * v

        return pos

    def get_colors(self, positions):
        colors = np.empty(len(positions), dtype=str)
        color_list = ['r', 'b', 'g', 'm', 'c', 'y']

        for i in range(len(positions)):
            color_list = np.roll(color_list, 1)
            colors[i] = color_list[0]

        return colors

    def main(self):
        self.load_network()
        graph = nx.DiGraph(nx.ego_graph(self.di_graph, SEED, radius=4))

        positions = nx.spring_layout(self.di_graph)

        colors = self.get_colors(positions)
        nx.draw_networkx_nodes(self.di_graph, positions, node_color=colors,
                                alpha=0.6)

        nx.draw_networkx_edges(self.di_graph, positions, width=0.5, alpha=0.5)
        # unresolved bug in the below for OSX
        #nx.draw_networkx_labels(self.di_graph, positions, font_size=9, alpha=0.7)
        # hack fix
        for string, pos in positions.iteritems():
            p1 = pos[0]
            p2 = pos[1]
            plt.text(p1, p2, string, fontsize=8)



